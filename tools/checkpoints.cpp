/**
 *  @file tools/checkpoints.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The verbs over checkpoints of every modality: their metadata, the slots of a raw drive
 *      holding them, and verified copies between files and slots.
 *
 *  A raw drive is cut into slots of 1 GiB from byte 0, and every file or volume on it starts on a
 *  slot boundary and covers whole slots; slot 0 holds no label. Mapping reads the first 4 KiB of
 *  each slot, which is live when it opens a SafeGraphs monolith whose header parses and whose
 *  declared extent fits the drive, or a UStore volume whose magic and header checksum hold. A live
 *  file covers ⌈length ÷ 1 GiB⌉ slots, which mapping skips, and every other slot is free. Owners
 *  come from the headers: the SafeGraphs metadata, or the UStore database UUID. Deleting a file
 *  zeroes its first block.
 */
#include <cstdint> // `std::uint64_t`, `std::uint32_t`
#include <cstdio>  // `stderr`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`
#include <cstring> // `std::memcmp`, `std::memcpy`, `std::memset`

#include <fcntl.h>     // `open`, `O_RDONLY`, `O_RDWR`, `O_DIRECT`, `O_EXCL`
#include <linux/fs.h>  // `BLKSSZGET`, `BLKGETSIZE64`
#include <sys/ioctl.h> // `ioctl`
#include <sys/stat.h>  // `fstat`, `stat`, `S_ISBLK`
#include <unistd.h>    // `pread`, `pwrite`, `close`, `fsync`

#include <algorithm>   // `std::min`, `std::max`
#include <array>       // `std::array`
#include <chrono>      // `std::chrono::steady_clock`
#include <iterator>    // `std::back_inserter`
#include <memory>      // `std::align`, `std::allocator`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include "usearch/usearch.h"
#include "usearch/utilities/blocking_files.hpp"    // `parse_file_address`
#include "usearch/utilities/nvme_device.hpp"       // `names_transport_id`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_header`, `safegraphs_read_metadata`
#include "usearch/utilities/types.hpp"             // `divide_round_up`, `round_up_to_multiple`, `little_endian`

#include "world.hpp" // `world_t`, `parse_verb`

namespace unum::usearch::tools {

using checkpoints_clock_t = std::chrono::steady_clock;

/** Bytes in one slot of a raw drive, where every file or volume starts and whose multiples it
 *  covers. */
static constexpr std::uint64_t slot_bytes_k = 1ull << 30;

/** Bytes mapping reads from the front of every slot. */
static constexpr std::size_t slot_probe_bytes_k = 4096;

/** Bytes one positional write or read of a copy moves. */
static constexpr std::size_t copy_chunk_bytes_k = 16ull << 20;

/** What a UStore volume opens with, packed: the magic "USTOREV1", the format version and the
 *  header's bytes as `u32`, the database UUID, the capacity, catalog, data, granularity and
 *  generation words as `u64`, then the CRC-32C of the header with that field zeroed. */
static constexpr std::string_view ustore_magic_k = "USTOREV1";
static constexpr std::uint32_t ustore_format_version_k = 1;
static constexpr std::size_t ustore_header_bytes_k = 96;
static constexpr std::size_t ustore_database_offset_k = 16;
static constexpr std::size_t ustore_capacity_offset_k = 32;
static constexpr std::size_t ustore_checksum_offset_k = 88;

/** A medium holding slots, open for positional I/O and closed whichever way a verb ends. */
struct slot_drive_t {
    int descriptor = -1;

    /** Bytes the medium spans. */
    std::uint64_t bytes = 0;

    /** The granularity of every transfer's offset and length: the logical block of a device opened
     *  for direct I/O, one byte otherwise. */
    std::uint64_t block_bytes = 1;

    slot_drive_t() = default;
    slot_drive_t(slot_drive_t const&) = delete;
    slot_drive_t& operator=(slot_drive_t const&) = delete;
    ~slot_drive_t() noexcept {
        if (descriptor >= 0) close(descriptor);
    }
};

/** A live file on a drive: its first slot, the bytes its header declares, and who owns it. */
struct slot_file_t {
    std::uint64_t first_slot = 0;
    std::uint64_t bytes = 0;
    std::string owner;
};

static usearch_bytes_view_t checkpoints_bytes(std::string_view text) noexcept {
    return {reinterpret_cast<usearch_byte_t const*>(text.data()), text.size()};
}

static double checkpoints_seconds_since(checkpoints_clock_t::time_point started) noexcept {
    return std::chrono::duration<double>(checkpoints_clock_t::now() - started).count();
}

/** Slots a live file of @p bytes covers, at least the one it starts in. */
static std::uint64_t slots_covered(std::uint64_t bytes) noexcept {
    return std::max<std::uint64_t>(1, utilities::divide_round_up(bytes, slot_bytes_k));
}

/** The noun a count of @p slots takes. */
static std::string_view slots_noun(std::uint64_t slots) noexcept { return slots == 1 ? "slot" : "slots"; }

/** A run of @p bytes inside @p storage, aligned for direct transfers. */
static std::span<usearch_byte_t> slot_aligned(std::vector<usearch_byte_t>& storage, std::size_t bytes) {
    storage.resize(bytes + slot_probe_bytes_k);
    void* aligned = storage.data();
    std::size_t space = storage.size();
    std::align(slot_probe_bytes_k, bytes, aligned, space);
    return {static_cast<usearch_byte_t*>(aligned), bytes};
}

/** Reads all of @p buffer at @p offset, retrying short reads; false on an error or at the end. */
static bool slot_read_at(int descriptor, std::span<usearch_byte_t> buffer, std::uint64_t offset) noexcept {
    while (!buffer.empty()) {
        ssize_t const read_bytes = pread(descriptor, buffer.data(), buffer.size(), static_cast<off_t>(offset));
        if (read_bytes <= 0) return false;
        buffer = buffer.subspan(static_cast<std::size_t>(read_bytes)), offset += read_bytes;
    }
    return true;
}

/** Writes all of @p buffer at @p offset, retrying short writes; false on an error. */
static bool slot_write_at(int descriptor, std::span<usearch_byte_t const> buffer, std::uint64_t offset) noexcept {
    while (!buffer.empty()) {
        ssize_t const written_bytes = pwrite(descriptor, buffer.data(), buffer.size(), static_cast<off_t>(offset));
        if (written_bytes <= 0) return false;
        buffer = buffer.subspan(static_cast<std::size_t>(written_bytes)), offset += written_bytes;
    }
    return true;
}

/** The little-endian integer at byte @p offset of @p bytes. */
template <typename integer_type_>
static integer_type_ slot_integer_at(std::span<usearch_byte_t const> bytes, std::size_t offset) noexcept {
    integer_type_ value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return utilities::little_endian(value);
}

/** The CRC-32C of @p bytes, the checksum a UStore volume header carries. */
static std::uint32_t slot_crc32c(std::span<usearch_byte_t const> bytes) noexcept {
    std::uint32_t remainder = ~0u;
    for (usearch_byte_t const byte : bytes) {
        remainder ^= byte;
        for (unsigned step = 0; step != 8; ++step)
            remainder = (remainder >> 1) ^ (0x82F63B78u & (0u - (remainder & 1u)));
    }
    return ~remainder;
}

/** Opens the medium at @p location under @p flags, for direct I/O when it is a block device and
 *  @p flags ask for it; false after printing why. */
static bool slot_open(std::string const& location, int flags, slot_drive_t& drive) {
    drive.descriptor = open(location.c_str(), flags | O_CLOEXEC, 0644);
    struct stat medium_stat;
    if (drive.descriptor < 0 || fstat(drive.descriptor, &medium_stat) != 0) {
        fmt::println(stderr, "cannot open {}", location);
        return false;
    }
    if (!S_ISBLK(medium_stat.st_mode)) {
        drive.bytes = static_cast<std::uint64_t>(medium_stat.st_size);
        return true;
    }
    int block_bytes = 0;
    if (ioctl(drive.descriptor, BLKGETSIZE64, &drive.bytes) != 0 || ioctl(drive.descriptor, BLKSSZGET, &block_bytes) ||
        block_bytes <= 0) {
        fmt::println(stderr, "cannot measure {}", location);
        return false;
    }
    if (flags & O_DIRECT) drive.block_bytes = static_cast<std::uint64_t>(block_bytes);
    return true;
}

/** Whether @p location names a block device. */
static bool slot_names_device(std::string const& location) noexcept {
    struct stat medium_stat;
    return stat(location.c_str(), &medium_stat) == 0 && S_ISBLK(medium_stat.st_mode);
}

/** Reads the SafeGraphs monolith opening at byte @p offset of @p drive, whose first bytes are
 *  @p probe, into @p file; false unless its header parses and its extent fits the drive. */
static bool slot_probe_safegraphs(slot_drive_t const& drive, std::uint64_t offset,
                                  std::span<usearch_byte_t const> probe, slot_file_t& file) {
    using prefix_t = utilities::safegraphs_prefix_t;
    if (std::memcmp(probe.data(), prefix_t::magic_k.data(), prefix_t::magic_k.size()) != 0) return false;
    std::uint64_t const header_length = slot_integer_at<std::uint64_t>(probe, prefix_t::magic_k.size());
    if (header_length > drive.bytes - offset - sizeof(prefix_t)) return false;
    std::size_t const header_bytes = static_cast<std::size_t>(sizeof(prefix_t) + header_length);
    std::vector<usearch_byte_t> storage;
    std::span<usearch_byte_t const> header = probe.first(std::min(header_bytes, probe.size()));
    if (header_bytes > probe.size()) {
        std::size_t const padded = utilities::round_up_to_multiple<std::size_t>(header_bytes, drive.block_bytes);
        std::span<usearch_byte_t> const longer = slot_aligned(storage, padded);
        if (padded > drive.bytes - offset || !slot_read_at(drive.descriptor, longer, offset)) return false;
        header = longer.first(header_bytes);
    }
    std::string metric, producer;
    auto per_custom_field = [&](utilities::safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metric = field.value;
        else if (field.key == "created_by") producer = field.value;
    };
    auto const [parsed, status] = utilities::safegraphs_read_header(
        utilities::bytes_of(header.data(), header.size()), std::allocator<utilities::byte_t> {}, per_custom_field);
    if (utilities::failed(status) || parsed.monolith_bytes() > drive.bytes - offset) return false;
    file.first_slot = offset / slot_bytes_k;
    file.bytes = parsed.monolith_bytes();
    file.owner = fmt::format("SafeGraphs of {} nodes of {}, metric \"{}\"", parsed.nodes(),
                             utilities::dtype_name(parsed.value.dtype), metric);
    if (!producer.empty()) fmt::format_to(std::back_inserter(file.owner), ", created by \"{}\"", producer);
    return true;
}

/** Reads the UStore volume opening at byte @p offset, whose first bytes are @p probe, into @p file;
 *  false unless its magic, version and header checksum hold. */
static bool slot_probe_ustore(std::uint64_t offset, std::span<usearch_byte_t const> probe, slot_file_t& file) {
    if (std::memcmp(probe.data(), ustore_magic_k.data(), ustore_magic_k.size()) != 0) return false;
    if (slot_integer_at<std::uint32_t>(probe, ustore_magic_k.size()) != ustore_format_version_k ||
        slot_integer_at<std::uint32_t>(probe, ustore_magic_k.size() + 4) != ustore_header_bytes_k)
        return false;
    std::array<usearch_byte_t, ustore_header_bytes_k> header;
    std::memcpy(header.data(), probe.data(), header.size());
    std::memset(header.data() + ustore_checksum_offset_k, 0, sizeof(std::uint32_t));
    if (slot_crc32c(header) != slot_integer_at<std::uint32_t>(probe, ustore_checksum_offset_k)) return false;
    std::string database;
    for (std::size_t byte_index = 0; byte_index != 16; ++byte_index)
        fmt::format_to(std::back_inserter(database), "{}{:02x}",
                       byte_index == 4 || byte_index == 6 || byte_index == 8 || byte_index == 10 ? "-" : "",
                       probe[ustore_database_offset_k + byte_index]);
    file.first_slot = offset / slot_bytes_k;
    file.bytes = slot_integer_at<std::uint64_t>(probe, ustore_capacity_offset_k);
    file.owner = fmt::format("UStore volume of database {}", database);
    return true;
}

/** Bytes a probe reads: the first 4 KiB, rounded up to the drive's block. */
static std::size_t slot_probe_length(slot_drive_t const& drive) noexcept {
    return utilities::round_up_to_multiple<std::size_t>(slot_probe_bytes_k, drive.block_bytes);
}

/** Reads the live file opening at byte @p offset of @p drive into @p file; false where none
 *  does. */
static bool slot_probe(slot_drive_t const& drive, std::uint64_t offset, std::span<usearch_byte_t> probe,
                       slot_file_t& file) {
    if (probe.size() > drive.bytes - std::min(offset, drive.bytes) || !slot_read_at(drive.descriptor, probe, offset))
        return false;
    return slot_probe_safegraphs(drive, offset, probe, file) || slot_probe_ustore(offset, probe, file);
}

/** Maps every live file on @p drive into @p files in slot order; false after printing a failed
 *  read. */
static bool slot_map(slot_drive_t const& drive, std::string_view location, std::vector<slot_file_t>& files) {
    std::uint64_t const slots = drive.bytes / slot_bytes_k;
    std::vector<usearch_byte_t> storage;
    std::span<usearch_byte_t> const probe = slot_aligned(storage, slot_probe_length(drive));
    for (std::uint64_t slot = 0; slot < slots;) {
        if (!slot_read_at(drive.descriptor, probe, slot * slot_bytes_k)) {
            fmt::println(stderr, "cannot read slot {} of {}", slot, location);
            return false;
        }
        slot_file_t file;
        if (!slot_probe_safegraphs(drive, slot * slot_bytes_k, probe, file) &&
            !slot_probe_ustore(slot * slot_bytes_k, probe, file)) {
            ++slot;
            continue;
        }
        slot += slots_covered(file.bytes);
        files.push_back(std::move(file));
    }
    return true;
}

/** The live file covering a slot of @p count from @p first, or null where they are all free. */
static slot_file_t const* slot_overlap(std::span<slot_file_t const> files, std::uint64_t first,
                                       std::uint64_t count) noexcept {
    for (slot_file_t const& file : files)
        if (file.first_slot < first + count && first < file.first_slot + slots_covered(file.bytes)) return &file;
    return nullptr;
}

/** The first slot of the first free run of @p count among @p slots, or @p slots where none is long
 *  enough. */
static std::uint64_t slot_free_run(std::span<slot_file_t const> files, std::uint64_t slots,
                                   std::uint64_t count) noexcept {
    std::uint64_t start = 0;
    for (slot_file_t const& file : files) {
        if (file.first_slot >= start + count) break;
        start = std::max(start, file.first_slot + slots_covered(file.bytes));
    }
    return start + count <= slots ? start : slots;
}

/** Opens the drive @p location names and maps it, refusing an SPDK address or a byte base. */
static bool slot_open_drive(std::string_view address, int flags, std::string& location, slot_drive_t& drive,
                            std::vector<slot_file_t>& files) {
    auto const [parsed, status] = utilities::parse_file_address(address);
    if (utilities::failed(status)) {
        fmt::println(stderr, "{} is no address", address);
        return false;
    }
    location = parsed.location;
    return slot_open(location, flags, drive) && slot_map(drive, location, files);
}

bool checkpoints_admit_save(std::string_view address) {
    // A drive bound to SPDK is out of the kernel's reach, and only the library maps it.
    if (utilities::names_transport_id(address)) return true;
    auto const [parsed, status] = utilities::parse_file_address(address);
    if (utilities::failed(status)) {
        fmt::println(stderr, "{} is no address", address);
        return false;
    }
    std::string const location {parsed.location};
    if (!slot_names_device(location)) return true;
    if (parsed.location.size() == address.size()) {
        fmt::println(
            stderr,
            "{} names a drive without a byte base, and no call sizes a save before it writes; " "name a free " "slot " "that " "usearch " "slots " "lists, " "like {}@<slot>GiB",
            address, location);
        return false;
    }
    if (parsed.byte_base % slot_bytes_k) {
        fmt::println(stderr, "{} starts off a slot boundary, a multiple of 1 GiB", address);
        return false;
    }
    slot_drive_t drive;
    std::vector<slot_file_t> files;
    if (!slot_open(location, O_RDONLY, drive) || !slot_map(drive, location, files)) return false;
    if (slot_file_t const* live = slot_overlap(files, parsed.byte_base / slot_bytes_k, 1)) {
        fmt::println(stderr, "{} is inside the {} at {}@{}GiB", address, live->owner, location, live->first_slot);
        return false;
    }
    return true;
}

int checkpoints_slots(world_t& world, int argc, char const* const* argv) {
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch slots <drive>", {}, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1 || utilities::names_transport_id(positionals[0])) {
        fmt::println(stderr, "a drive the kernel's nvme driver holds is required, like /dev/nvme1n1");
        return EXIT_FAILURE;
    }
    std::string location;
    slot_drive_t drive;
    std::vector<slot_file_t> files;
    if (!slot_open_drive(positionals[0], O_RDONLY, location, drive, files)) return EXIT_FAILURE;

    std::uint64_t const slots = drive.bytes / slot_bytes_k;
    fmt::println("{}: {} bytes in {} slots of 1 GiB, {} live", location, drive.bytes, slots, files.size());
    std::uint64_t next = 0;
    for (slot_file_t const& file : files) {
        if (file.first_slot > next)
            fmt::println("{}@{}GiB: free, {} {}", location, next, file.first_slot - next,
                         slots_noun(file.first_slot - next));
        std::uint64_t const covered = slots_covered(file.bytes);
        fmt::println("{}@{}GiB: {}, {} bytes over {} {}", location, file.first_slot, file.owner, file.bytes, covered,
                     slots_noun(covered));
        next = file.first_slot + covered;
    }
    if (next < slots) fmt::println("{}@{}GiB: free, {} {}", location, next, slots - next, slots_noun(slots - next));
    return EXIT_SUCCESS;
}

int checkpoints_copy(world_t& world, int argc, char const* const* argv) {
    std::array<std::string_view, 2> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch copy <from> <to>", {}, positionals,
                                                   given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    std::string_view const from = positionals[0], to = positionals[1];
    if (given != 2 || utilities::names_transport_id(from) || utilities::names_transport_id(to)) {
        fmt::println(stderr, "a source and a destination the kernel reaches are required");
        return EXIT_FAILURE;
    }
    auto const [source_address, source_status] = utilities::parse_file_address(from);
    auto const [target_address, target_status] = utilities::parse_file_address(to);
    if (utilities::failed(source_status) || utilities::failed(target_status)) {
        fmt::println(stderr, "{} is no address", utilities::failed(source_status) ? from : to);
        return EXIT_FAILURE;
    }

    // A drive holds files back to back, so its source is the live file its header bounds.
    std::string const source_location {source_address.location};
    slot_drive_t source;
    if (!slot_open(source_location, O_RDONLY, source)) return EXIT_FAILURE;
    std::uint64_t const source_base = source_address.byte_base;
    std::uint64_t length = source.bytes - std::min(source_base, source.bytes);
    if (slot_names_device(source_location)) {
        std::vector<usearch_byte_t> storage;
        slot_file_t file;
        if (!slot_probe(source, source_base, slot_aligned(storage, slot_probe_length(source)), file)) {
            fmt::println(stderr, "{} holds no SafeGraphs monolith or UStore volume", from);
            return EXIT_FAILURE;
        }
        length = file.bytes;
    }

    std::string const target_location {target_address.location};
    slot_drive_t target;
    std::uint64_t target_base = target_address.byte_base;
    if (slot_names_device(target_location)) {
        // An exclusive open of a block device fails while anything holds it mounted.
        std::vector<slot_file_t> files;
        if (!slot_open(target_location, O_RDWR | O_DIRECT | O_EXCL, target) ||
            !slot_map(target, target_location, files))
            return EXIT_FAILURE;
        std::uint64_t const slots = target.bytes / slot_bytes_k, needed = slots_covered(length);
        if (target_address.location.size() == to.size())
            target_base = slot_free_run(files, slots, needed) * slot_bytes_k;
        if (target_base % slot_bytes_k || target_base / slot_bytes_k + needed > slots) {
            fmt::println(stderr, "{} {} from byte {} of {} would sit off its slot boundaries or past its {} slots",
                         needed, slots_noun(needed), target_base, target_location, slots);
            return EXIT_FAILURE;
        }
        if (slot_file_t const* live = slot_overlap(files, target_base / slot_bytes_k, needed)) {
            fmt::println(stderr, "{} {} from {}@{}GiB would overlap the {} at {}@{}GiB", needed, slots_noun(needed),
                         target_location, target_base / slot_bytes_k, live->owner, target_location, live->first_slot);
            return EXIT_FAILURE;
        }
    }
    else if (!slot_open(target_location, O_RDWR | O_CREAT | (target_base ? 0 : O_TRUNC), target)) return EXIT_FAILURE;

    // Direct transfers want block-aligned memory, so two chunks are carved from one aligned span.
    std::vector<usearch_byte_t> storage;
    std::span<usearch_byte_t> const chunks = slot_aligned(storage, 2 * copy_chunk_bytes_k);
    std::span<usearch_byte_t> const staged = chunks.first(copy_chunk_bytes_k);
    std::span<usearch_byte_t> const read_back = chunks.subspan(copy_chunk_bytes_k);
    checkpoints_clock_t::time_point started = checkpoints_clock_t::now();
    for (std::uint64_t offset = 0; offset < length; offset += copy_chunk_bytes_k) {
        std::size_t const chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(copy_chunk_bytes_k, length - offset));
        std::size_t const padded = utilities::round_up_to_multiple<std::size_t>(chunk, target.block_bytes);
        std::memset(staged.data() + chunk, 0, padded - chunk);
        if (!slot_read_at(source.descriptor, staged.first(chunk), source_base + offset)) {
            fmt::println(stderr, "cannot read {} bytes at {} of {}", chunk, source_base + offset, from);
            return EXIT_FAILURE;
        }
        if (!slot_write_at(target.descriptor, staged.first(padded), target_base + offset)) {
            fmt::println(stderr, "cannot write {} bytes at {} of {}", padded, target_base + offset, target_location);
            return EXIT_FAILURE;
        }
    }
    if (fsync(target.descriptor) != 0) {
        fmt::println(stderr, "cannot flush {}", target_location);
        return EXIT_FAILURE;
    }
    double const written_seconds = checkpoints_seconds_since(started);
    fmt::println("copied {} to {} at byte {}: {} bytes in {:.1f} s, {:.2f} GB/s", from, target_location, target_base,
                 length, written_seconds, length / written_seconds / 1e9);

    started = checkpoints_clock_t::now();
    for (std::uint64_t offset = 0; offset < length; offset += copy_chunk_bytes_k) {
        std::size_t const chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(copy_chunk_bytes_k, length - offset));
        std::size_t const padded = utilities::round_up_to_multiple<std::size_t>(chunk, target.block_bytes);
        if (!slot_read_at(source.descriptor, staged.first(chunk), source_base + offset) ||
            !slot_read_at(target.descriptor, read_back.first(padded), target_base + offset)) {
            fmt::println(stderr, "cannot read back {} bytes at {}", chunk, offset);
            return EXIT_FAILURE;
        }
        if (std::memcmp(staged.data(), read_back.data(), chunk) != 0) {
            fmt::println(stderr, "{} differs from {} within the {} bytes at {}", target_location, from, chunk, offset);
            return EXIT_FAILURE;
        }
    }
    fmt::println("verified {} bytes in {:.1f} s", length, checkpoints_seconds_since(started));
    return EXIT_SUCCESS;
}

/** The modalities a checkpoint's metric token tells apart. */
enum class checkpoints_modality_t { dense_k, sequence_k, shapes_k, geospatial_k };

/** The modality whose engine records @p metric, by the tokens its checkpoints carry; dense for the
 *  rest, whose tokens are NumKong's kernel names. */
static checkpoints_modality_t checkpoints_modality_of(std::string_view metric) noexcept {
    if (metric == "levenshtein" || metric == "levenshtein-utf8" || metric == "overlap")
        return checkpoints_modality_t::sequence_k;
    if (metric == "chamfer" || metric == "hausdorff" || metric == "gromov_wasserstein" ||
        metric == "wasserstein_procrustes")
        return checkpoints_modality_t::shapes_k;
    if (metric == "chord") return checkpoints_modality_t::geospatial_k;
    return checkpoints_modality_t::dense_k;
}

/** Prints a failed call's status and message; true when it failed. */
static bool checkpoints_failed(usearch_outcome_t const& outcome, std::string_view address) {
    if (outcome.status == usearch_status_success_k) return false;
    fmt::println(stderr, "reading {} failed with {}: {}", address, usearch_status_name(outcome.status),
                 outcome.message ? outcome.message : "no message");
    return true;
}

int checkpoints_metadata(world_t& world, int argc, char const* const* argv) {
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch metadata <address>", {}, positionals,
                                                   given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    std::string_view const address = positionals[0];
    if (given != 1 || utilities::names_transport_id(address)) {
        fmt::println(stderr, "a monolith or manifest the kernel reaches is required");
        return EXIT_FAILURE;
    }
    auto const [header, header_status] = utilities::safegraphs_read_metadata(address);
    if (utilities::failed(header_status)) {
        fmt::println(stderr, "{} holds no SafeGraphs header or manifest", address);
        return EXIT_FAILURE;
    }
    std::string_view const metric = header.metric.view();
    fmt::println("{}: {} nodes in {} parts, metric \"{}\", {} keys", address, header.nodes, header.parts, metric,
                 utilities::dtype_name(header.key_dtype));

    usearch_outcome_t outcome {};
    usearch_bytes_view_t const path = checkpoints_bytes(address);
    switch (checkpoints_modality_of(metric)) {
    case checkpoints_modality_t::dense_k: {
        usearch_dense_metadata_t metadata {};
        usearch_dense_metadata_read_t read {.outcome = &outcome, .path = path, .metadata = &metadata};
        usearch_dense_metadata_read(&read);
        if (checkpoints_failed(outcome, address)) return EXIT_FAILURE;
        fmt::println("dense: {} metric, {} dimensions of {}, connectivity {}",
                     usearch_dense_metric_name(metadata.metric), metadata.dimensions,
                     usearch_scalar_name(metadata.quantization), metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::sequence_k: {
        usearch_sequence_metadata_t metadata {};
        usearch_sequence_metadata_read_t read {.outcome = &outcome, .path = path, .metadata = &metadata};
        usearch_sequence_metadata_read(&read);
        if (checkpoints_failed(outcome, address)) return EXIT_FAILURE;
        fmt::println("sequence: connectivity {}", metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::shapes_k: {
        usearch_shapes_metadata_t metadata {};
        usearch_shapes_metadata_read_t read {.outcome = &outcome, .path = path, .metadata = &metadata};
        usearch_shapes_metadata_read(&read);
        if (checkpoints_failed(outcome, address)) return EXIT_FAILURE;
        fmt::println("shapes: {} points of {}, connectivity {}", metadata.points,
                     usearch_scalar_name(metadata.quantization), metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::geospatial_k: {
        usearch_geospatial_metadata_t metadata {};
        usearch_geospatial_metadata_read_t read {.outcome = &outcome, .path = path, .metadata = &metadata};
        usearch_geospatial_metadata_read(&read);
        if (checkpoints_failed(outcome, address)) return EXIT_FAILURE;
        fmt::println("geospatial: connectivity {}", metadata.connectivity);
        return EXIT_SUCCESS;
    }
    }
    return EXIT_FAILURE;
}

} // namespace unum::usearch::tools
