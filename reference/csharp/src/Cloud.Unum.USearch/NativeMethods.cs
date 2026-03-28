using System.Runtime.InteropServices;

using usearch_index_t = System.IntPtr;
using usearch_key_t = System.UInt64;
using usearch_distance_t = System.Single;
using usearch_error_t = System.IntPtr;
using size_t = System.UIntPtr;
using void_ptr_t = System.IntPtr;
using usearch_key_ptr_t = System.IntPtr;
using usearch_distance_ptr_t = System.IntPtr;

namespace Cloud.Unum.USearch;

internal static class NativeMethods
{
    private const string LibraryName = "libusearch_c";

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern usearch_index_t usearch_init(ref IndexOptions options, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_free(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_save(usearch_index_t index, [MarshalAs(UnmanagedType.LPStr)] string path, out usearch_error_t error);

    [DllImport(LibraryName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_load(usearch_index_t index, [MarshalAs(UnmanagedType.LPStr)] string path, out usearch_error_t error);

    [DllImport(LibraryName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_view(usearch_index_t index, [MarshalAs(UnmanagedType.LPStr)] string path, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_size(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_capacity(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_dimensions(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_connectivity(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_reserve(usearch_index_t index, size_t capacity, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_add(
        usearch_index_t index,
        usearch_key_t key,
        void_ptr_t vector,
        ScalarKind vector_kind,
        out usearch_error_t error
    );


    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool usearch_contains(usearch_index_t index, usearch_key_t key, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_count(usearch_index_t index, usearch_key_t key, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_search(
        usearch_index_t index,
        void_ptr_t query_vector,
        ScalarKind query_kind,
        size_t count,
        usearch_key_ptr_t found_keys,
        usearch_distance_ptr_t found_distances,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_get(
        usearch_index_t index,
        usearch_key_t key,
        size_t count,
        void_ptr_t vector,
        ScalarKind vector_kind,
        out usearch_error_t error
    );


    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_remove(usearch_index_t index, usearch_key_t key, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern size_t usearch_rename(usearch_index_t index, usearch_key_t key_from, usearch_key_t key_to, out usearch_error_t error);
}
