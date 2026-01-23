//
//  Test.swift
//
//
//  Created by Ash Vardanian on 5/11/23.
//

import Foundation
import USearch
import XCTest

@available(iOS 13, macOS 10.15, tvOS 13.0, watchOS 6.0, visionOS 1.0, *)
class Test: XCTestCase {
    func testUnit() throws {
        let index = try USearchIndex.make(
            metric: USearchMetric.l2sq,
            dimensions: 4,
            connectivity: 8,
            quantization: USearchScalar.f32
        )
        let vectorA: [Float32] = [0.3, 0.5, 1.2, 1.4]
        let vectorB: [Float32] = [0.4, 0.2, 1.2, 1.1]
        try index.reserve(2)

        // Adding a slice
        try index.add(key: 42, vector: vectorA[...])

        // Adding a vector
        try index.add(key: 43, vector: vectorB)

        let results = try index.search(vector: vectorA, count: 10)
        XCTAssertEqual(results.0[0], 42)

        let fetched: [[Float]]? = try index.get(key: 42)
        XCTAssertEqual(fetched?[0], vectorA)

        XCTAssertTrue(try index.contains(key: 42))
        XCTAssertEqual(try index.count(key: 42), 1)
        XCTAssertEqual(try index.count(key: 49), 0)
        try _ = index.rename(from: 42, to: 49)
        XCTAssertEqual(try index.count(key: 49), 1)

        let fetched_renamed: [[Float]]? = try index.get(key: 49)
        XCTAssertEqual(fetched_renamed?[0], vectorA)
        let stale: [[Float]]? = try index.get(key: 42)
        XCTAssertNil(stale)

        try _ = index.remove(key: 49)
        XCTAssertEqual(try index.count(key: 49), 0)
    }

    func testUnitMulti() throws {
        let index = try USearchIndex.make(
            metric: USearchMetric.l2sq,
            dimensions: 4,
            connectivity: 8,
            quantization: USearchScalar.f32,
            multi: true
        )
        let vectorA: [Float32] = [0.3, 0.5, 1.2, 1.4]
        let vectorB: [Float32] = [0.4, 0.2, 1.2, 1.1]
        try index.reserve(2)

        // Adding a slice
        try index.add(key: 42, vector: vectorA[...])

        // Adding a vector
        try index.add(key: 42, vector: vectorB)

        let results = try index.search(vector: vectorA, count: 10)
        XCTAssertEqual(results.0[0], 42)

        let fetched: [[Float]]? = try index.get(key: 42, count: 2)
        XCTAssertEqual(fetched?.contains(vectorA), true)
        XCTAssertEqual(fetched?.contains(vectorB), true)

        XCTAssertTrue(try index.contains(key: 42))
        XCTAssertEqual(try index.count(key: 42), 2)
        XCTAssertEqual(try index.count(key: 49), 0)
        _ = try index.rename(from: 42, to: 49)
        XCTAssertEqual(try index.count(key: 49), 2)

        let refetched: [[Float]]? = try index.get(key: 49, count: 2)
        XCTAssertEqual(refetched?.contains(vectorA), true)
        XCTAssertEqual(refetched?.contains(vectorB), true)
        let stale: [[Float]]? = try index.get(key: 42)
        XCTAssertNil(stale)

        try _ = index.remove(key: 49)
        XCTAssertEqual(try index.count(key: 49), 0)
    }

    func testIssue399() throws {
        let index = try USearchIndex.make(
            metric: USearchMetric.l2sq,
            dimensions: 1,
            connectivity: 8,
            quantization: USearchScalar.f32
        )
        try index.reserve(3)

        // add 3 entries then ensure all 3 are returned
        try index.add(key: 1, vector: [1.1])
        try index.add(key: 2, vector: [2.1])
        try index.add(key: 3, vector: [3.1])
        try XCTAssertEqual(index.count, 3)
        XCTAssertEqual(try index.search(vector: [1.0], count: 3).0, [1, 2, 3])  // works 😎

        // replace second-added entry then ensure all 3 are still returned
        try _ = index.remove(key: 2)
        try index.add(key: 2, vector: [2.2])
        try XCTAssertEqual(index.count, 3)
        XCTAssertEqual(try index.search(vector: [1.0], count: 3).0, [1, 2, 3])  // works 😎

        // replace first-added entry then ensure all 3 are still returned
        try _ = index.remove(key: 1)
        try index.add(key: 1, vector: [1.2])
        let afterReplacingInitial = try index.search(vector: [1.0], count: 3).0
        try XCTAssertEqual(index.count, 3)
        XCTAssertEqual(afterReplacingInitial, [1, 2, 3])  // v2.11.7 fails with "[1] != [1, 2, 3]" 😨
    }

    func testFilteredSearchSingle() throws {
        let index = try USearchIndex.make(
            metric: USearchMetric.l2sq,
            dimensions: 1,
            connectivity: 8,
            quantization: USearchScalar.f32
        )
        try index.reserve(3)

        // add 3 entries
        try index.add(key: 1, vector: [1.1])
        try index.add(key: 2, vector: [2.1])
        try index.add(key: 3, vector: [3.1])
        try XCTAssertEqual(index.count, 3)

        // filter which accepts all keys:
        XCTAssertEqual(
            try index.filteredSearch(vector: [1.0], count: 3) {
                key in true
            }.0,
            [1, 2, 3]
        )  // works 😎

        // filter which rejects all keys:
        XCTAssertEqual(
            try index.filteredSearch(vector: [1.0], count: 3) {
                key in false
            }.0,
            []
        )  // works 😎

        // filter function accepts a set of keys passed in through a capture.
        let acceptedKeys: [USearchKey] = [1, 2]
        XCTAssertEqual(
            try index.filteredSearch(vector: [1.0], count: 3) {
                key in acceptedKeys.contains(key)
            }.0,
            acceptedKeys
        )  // works 😎

        // filter function accepts a set of keys passed in through a capture,
        // and also adheres to the count.
        XCTAssertEqual(
            try index.filteredSearch(vector: [1.0], count: 1) {
                key in key > 1
            }.0,
            [2]
        )  // works 😎
        XCTAssertEqual(
            try index.filteredSearch(vector: [1.0], count: 2) {
                key in key > 1
            }.0,
            [2, 3]
        )  // works 😎
    }

    func testFilteredSearchDouble() throws {
        let index = try USearchIndex.make(
            metric: USearchMetric.l2sq,
            dimensions: 1,
            connectivity: 8,
            quantization: USearchScalar.f64
        )
        try index.reserve(3)

        // add 3 entries
        try index.add(key: 1, vector: [Float64(1.1)])
        try index.add(key: 2, vector: [Float64(2.1)])
        try index.add(key: 3, vector: [Float64(3.1)])
        try XCTAssertEqual(index.count, 3)

        // filter which accepts all keys:
        XCTAssertEqual(
            try index.filteredSearch(vector: [Float64(1.0)], count: 3) {
                key in true
            }.0,
            [1, 2, 3]
        )  // works 😎

        // filter which rejects all keys:
        XCTAssertEqual(
            try index.filteredSearch(vector: [Float64(1.0)], count: 3) {
                key in false
            }.0,
            []
        )  // works 😎

        // filter function accepts a set of keys passed in through a capture.
        let acceptedKeys: [USearchKey] = [1, 2]
        XCTAssertEqual(
            try index.filteredSearch(vector: [Float64(1.0)], count: 3) {
                key in acceptedKeys.contains(key)
            }.0,
            acceptedKeys
        )  // works 😎

        // filter function accepts a set of keys passed in through a capture,
        // and also respects the count.
        XCTAssertEqual(
            try index.filteredSearch(vector: [Float64(1.0)], count: 1) {
                key in key > 1
            }.0,
            [2]
        )  // works 😎
        XCTAssertEqual(
            try index.filteredSearch(vector: [Float64(1.0)], count: 2) {
                key in key > 1
            }.0,
            [2, 3]
        )  // works 😎
    }
}
