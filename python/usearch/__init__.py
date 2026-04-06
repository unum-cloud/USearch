import ctypes
import os
import platform
import sys
import urllib.request
import warnings
from urllib.error import HTTPError

#! Load NumKong before the USearch compiled module
#! We can't just use the `import numkong` as on Linux and Windows (unlike MacOS),
#! the symbols are not automatically loaded into the global namespace.
try:
    import numkong

    # Cross-platform check for Windows
    if sys.platform == "win32":
        # Add the directory where the `.dll` is located
        dll_directory = os.path.dirname(numkong.__file__)
        os.add_dll_directory(dll_directory)

        # Load NumKong library using `ctypes` without `RTLD_GLOBAL`
        numkong_lib = ctypes.CDLL(numkong.__file__)

    else:
        # Non-Windows: Use `RTLD_GLOBAL` for Unix-based systems (Linux/macOS)
        numkong_lib = ctypes.CDLL(numkong.__file__, mode=ctypes.RTLD_GLOBAL)

except ImportError:
    pass  # If the user doesn't want NumKong, we assume they know what they're doing


from usearch.compiled import (  # type: ignore[import-not-found]
    # Default values:
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
    USES_NUMKONG,
    USES_NUMKONG_DYNAMIC_DISPATCH,
    # Dependencies:
    USES_OPENMP,
    USES_SIMSIMD,
    USES_SIMSIMD_DYNAMIC_DISPATCH,
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
    hardware_acceleration_available,
    # Hardware capabilities:
    hardware_acceleration_compiled,
)

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"


class BinaryManager:
    def __init__(self, version: str | None = None):
        if version is None:
            version = __version__
        self.version = version or __version__

    @staticmethod
    def determine_download_dir() -> str:
        # Check if running within a virtual environment
        virtual_env = os.getenv("VIRTUAL_ENV")
        if virtual_env:
            # Use a subdirectory within the virtual environment for binaries
            return os.path.join(virtual_env, "bin", "usearch_binaries")
        else:
            # Fallback to a directory in the user's home folder
            home_dir = os.path.expanduser("~")
            return os.path.join(home_dir, ".usearch", "binaries")

    @staticmethod
    def determine_download_url(version: str, filename: str) -> str:
        base_url = "https://github.com/unum-cloud/USearch/releases/download"
        url = f"{base_url}/v{version}/{filename}"
        return url

    def get_binary_name(self) -> tuple[str, str]:
        version = self.version
        os_map = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}
        arch_map = {
            "x86_64": "amd64" if platform.system() != "Darwin" else "x86_64",
            "AMD64": "amd64",
            "arm64": "arm64",
            "aarch64": "arm64",
            "x86": "x86",
        }
        os_part = os_map.get(platform.system(), "")
        arch = platform.machine()
        arch_part = arch_map.get(arch, "")
        extension = {"Linux": "so", "Windows": "dll", "Darwin": "dylib"}.get(platform.system(), "")
        source_filename = f"usearch_sqlite_{os_part}_{arch_part}_{version}.{extension}"
        target_filename = f"usearch_sqlite.{extension}"
        return source_filename, target_filename

    def sqlite_found_or_downloaded(self) -> str | None:
        """
        Attempts to locate the pre-installed `usearch_sqlite` binary.
        If not found, downloads it from GitHub.

        Returns:
            The path to the binary if found or downloaded, otherwise None.
        """
        # Search local directories
        local_dirs = ["build", "build_artifacts", "build_release", "build_debug"]
        source_filename, target_filename = self.get_binary_name()

        # Check local development directories first
        for local_dir in local_dirs:
            local_path = os.path.join(local_dir, target_filename)
            if os.path.exists(local_path):
                path_wout_extension, _, _ = local_path.rpartition(".")
                return path_wout_extension

            # Most build systems on POSIX would prefix the library name with "lib"
            local_path = os.path.join(local_dir, "lib" + target_filename)
            if os.path.exists(local_path):
                path_wout_extension, _, _ = local_path.rpartition(".")
                return path_wout_extension

        # Check local installation directories, in case the build is already installed
        download_dir = self.determine_download_dir()
        local_path = os.path.join(download_dir, target_filename)
        if not os.path.exists(local_path):
            # If not found locally, warn the user and download from GitHub
            warnings.warn("Will download `usearch_sqlite` binary from GitHub.", UserWarning)
            try:
                source_url = self.determine_download_url(self.version, source_filename)
                os.makedirs(download_dir, exist_ok=True)
                urllib.request.urlretrieve(source_url, local_path)
            except HTTPError as e:
                # If the download fails due to HTTPError (e.g., 404 Not Found), like a missing lib version
                if e.code == 404:
                    warnings.warn(f"Download failed: {e.url} could not be found.", UserWarning)
                else:
                    warnings.warn(
                        f"Download failed with HTTP error: {e.code} {e.reason}",
                        UserWarning,
                    )
                return None

        # Handle the case where binary_path does not exist after supposed successful download
        if os.path.exists(local_path):
            path_wout_extension, _, _ = local_path.rpartition(".")
            return path_wout_extension
        else:
            warnings.warn("Failed to download `usearch_sqlite` binary from GitHub.", UserWarning)
            return None


def sqlite_path(version: str | None = None) -> str:
    manager = BinaryManager(version=version)
    result = manager.sqlite_found_or_downloaded()
    if result is None:
        raise FileNotFoundError("Failed to find or download `usearch_sqlite` binary.")
    return result
