// Package usearch provides Go bindings for USearch.
//
// The package links the precompiled engine, so building it compiles no index. Status, Key,
// Scalar, RowsView, Executor and Arena serve every modality; the dense modality carries a
// Dense prefix, leaving room for the modalities that follow.
//
//	options := usearch.DefaultDenseOptions()
//	options.Dimensions = 768
//	options.Quantization = usearch.ScalarI8
//
//	index, err := usearch.DenseInitRam(options, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Close()
//
//	if err := index.Reserve(1_000_000); err != nil {
//	    log.Fatal(err)
//	}
//
// File: golang/usearch.go
// Author: Ash Vardanian
// Date: October 30, 2026
package usearch
