//
//  Util.swift
//  USearch
//
//  Created by Dan Palmer on 09/02/2025.
//

import USearchC

func throwing<T>(_ fn: (inout UnsafeMutablePointer<usearch_error_t?>) -> T) throws -> T {
    // Allocate and initialize the pointer to nil.
    var err = UnsafeMutablePointer<usearch_error_t?>.allocate(capacity: 1)
    err.initialize(to: nil)

    // Ensure the allocated memory is deallocated when done.
    defer {
        err.deinitialize(count: 1)
        err.deallocate()
    }

    let result = fn(&err)
    if let errorCString = err.pointee {
        throw USearchError.fromErrorString(String(cString: errorCString))
    }
    return result
}

func pointer<T>(_ value: T) -> UnsafeMutablePointer<T> {
    let ptr = UnsafeMutablePointer<T>.allocate(capacity: 1)
    ptr.initialize(to: value)
    return ptr
}

class FilterWrapper {
    let filter: USearchFilterFn

    init(_ filter: @escaping USearchFilterFn) {
        self.filter = filter
    }
}

func filteredSearchGeneric<T>(
    _ index: usearch_index_t,
    vector: UnsafePointer<T>,
    count: UInt32,
    quantization: USearchScalar,
    filter: @escaping USearchFilterFn,
    keys: UnsafeMutablePointer<USearchKey>?,
    distances: UnsafeMutablePointer<Float32>?
) throws -> UInt32 {
    let filterBlock: (@convention(c) (usearch_key_t, UnsafeMutableRawPointer?) -> Int32) = { (key, state) in
        let wrapper = Unmanaged<FilterWrapper>.fromOpaque(state!).takeUnretainedValue()
        return wrapper.filter(key) ? 1 : 0
    }

    let unmanagedFilter = Unmanaged.passRetained(FilterWrapper(filter))
    let filterState = UnsafeMutableRawPointer(unmanagedFilter.toOpaque())

    let found = try throwing {
        let ret = usearch_filtered_search(
            index,
            vector,
            quantization.toNative(),
            Int(count),
            filterBlock,
            filterState,
            keys,
            distances,
            $0
        )
        unmanagedFilter.release()
        return ret
    }

    return UInt32(found)
}
