# ClusterR AI Coding Instructions

## Project Overview
ClusterR is an R package providing high-performance clustering algorithms including Gaussian Mixture Models, K-means, Mini-batch K-means, K-medoids, and Affinity Propagation. The package leverages RcppArmadillo for computational efficiency and provides a unique C++ linkage system for other R packages.

## Architecture & Core Components

### R/C++ Integration Pattern
- **Primary R interface**: Functions in `R/clustering_functions.R` provide user-facing API
- **C++ implementation**: Core algorithms in `src/` with auto-generated `RcppExports.cpp`
- **Header exposure**: `inst/include/ClusterRHeader.h` and `inst/include/affinity_propagation.h` expose C++ classes for external package linking
- **Namespace pattern**: C++ code uses `clustR` namespace with `ClustHeader` class containing all core methods

### Key Files for Understanding Architecture
- `R/clustering_functions.R`: Main R functions with consistent `tryCatch_*` error handling pattern
- `inst/include/ClusterRHeader.h`: Complete C++ API (3648+ lines) - THE authoritative reference for C++ integration
- `src/RcppExports.cpp`: Auto-generated Rcpp bindings (never edit directly)
- `NAMESPACE`: Exports following pattern: main algorithms + predict/print S3 methods

## Critical Development Workflows

### Building & Testing
```r
# Build package (R development)
devtools::document()    # Updates NAMESPACE from roxygen comments
devtools::build()       # Creates package bundle
devtools::check()       # Runs R CMD check

# Test specific algorithm families
testthat::test_file("tests/testthat/test-kmeans.R")
testthat::test_file("tests/testthat/test-gmm.R")
```

### C++ Compilation Setup
- **Makevars**: Uses OpenMP, Armadillo 64-bit words, requires LAPACK/BLAS
- **Key flags**: `-DARMA_64BIT_WORD -DARMA_USE_CURRENT` for Armadillo compatibility
- **Include paths**: `-I../inst/include` for internal headers

## Armadillo C++ Library Integration

ClusterR extensively uses RcppArmadillo for high-performance linear algebra operations. Understanding Armadillo patterns is crucial for working with this codebase.

### Core Armadillo Data Types & Usage Patterns
```cpp
// Primary matrix/vector types used throughout
arma::mat data;           // Double matrix (most common)
arma::vec clusters;       // Column vector
arma::rowvec centroids;   // Row vector  
arma::uvec indices;       // Unsigned integer vector for indices
arma::cube tensor;        // 3D array (used in some algorithms)
```

### Memory Management & Initialization Patterns
```cpp
// Zero/ones initialization (extensive use of arma::fill)
arma::vec count_clust(n_elem, arma::fill::zeros);
arma::mat centroids(clusters, n_cols, arma::fill::ones);
arma::rowvec distances(n_rows, arma::fill::randu);  // Random uniform

// Efficient matrix operations without copying
arma::mat& data_ref = data;  // Reference to avoid copying large matrices
```

### Type Conversions & Data Flow
```cpp
// R to Armadillo (automatic via RcppArmadillo)
// Rcpp::NumericMatrix -> arma::mat (seamless conversion)

// Armadillo internal conversions (common pattern)
arma::rowvec row_data = arma::conv_to<arma::rowvec>::from(data.row(i));
arma::uvec indices = arma::conv_to<arma::uvec>::from(medoids_vec.row(0));

// Back to R objects
return Rcpp::wrap(result_matrix);  // Armadillo -> R automatic
```

### OpenMP Parallelization Patterns
```cpp
// Thread setup (standard pattern across algorithms)
omp_set_num_threads(threads);

// Parallel loops with shared/private variable specifications
#pragma omp parallel for schedule(static) shared(data, centroids, clusters) private(i,j)
for (unsigned int i = 0; i < n_rows; i++) {
  // Atomic operations for thread-safe writes
  #pragma omp atomic write
  clusters[i] = best_cluster;
}
```

### Performance-Critical Armadillo Operations
- **Matrix operations**: Prefer `.t()` (transpose), `.i()` (inverse), element-wise ops
- **Distance calculations**: Extensive use of `arma::norm()`, `arma::dot()`, squared distances
- **Statistical functions**: `arma::mean()`, `arma::var()`, `arma::cov()` for centroid updates
- **Indexing**: `arma::find()`, `arma::unique()`, `arma::sort_index()` for cluster operations
- **Random sampling**: `arma::randu()`, `arma::randn()` for initialization algorithms

### Memory Efficiency Considerations
- **In-place operations**: Many functions use `arma::mat&` references to avoid copying
- **64-bit indexing**: `-DARMA_64BIT_WORD` enables large dataset support
- **LAPACK/BLAS backend**: Leverages optimized linear algebra libraries
- **OpenMP scaling**: Algorithms designed for multi-core parallelization

### Common Armadillo Gotchas in ClusterR
- **Matrix orientation**: Careful attention to row vs column operations (`data.row(i)` vs `data.col(j)`)
- **Index base**: Armadillo uses 0-based indexing (C++ style) vs R's 1-based
- **Element access**: `.at(i,j)` for bounds checking, `(i,j)` for performance
- **Submatrix operations**: `.rows(start, end)`, `.cols(start, end)` for efficient slicing

## Project-Specific Conventions

### Function Naming Patterns
- **Main algorithms**: `GMM()`, `KMeans_arma()`, `MiniBatchKmeans()`, `Cluster_Medoids()`, `AP_affinity_propagation()`
- **Optimization functions**: `Optimal_Clusters_*()` pattern for finding optimal cluster numbers
- **Error handling**: `tryCatch_*()` wrapper functions that handle Armadillo exceptions
- **S3 methods**: `predict.*Cluster()` and `print.*Cluster()` for each algorithm type

### Parameter Conventions
- **Distance modes**: `"eucl_dist"`, `"maha_dist"` standardized across algorithms
- **Seed modes**: `"static_subset"`, `"random_subset"`, `"static_spread"`, `"random_spread"`
- **Verbose output**: Boolean `verbose` parameter standard across all functions
- **Threading**: `threads` parameter for OpenMP parallelization where applicable

### Error Handling Pattern
```r
# Standard tryCatch wrapper pattern used throughout
tryCatch_ALGORITHM <- function(data, ...) {
  if (!is.matrix(data) && !is.data.frame(data)) {
    stop("data must be matrix or data.frame")
  }
  # Validate parameters...
  res = ALGORITHM_cpp(data, ...)  # Call C++ implementation
  # Post-process results...
}
```

## External Package Integration
The package provides a unique **LinkingTo** system for C++ code reuse:

### For packages wanting to use ClusterR C++ functions:
1. Add `LinkingTo: ClusterR` to DESCRIPTION
2. Include headers: `#include <ClusterRHeader.h>` and `#include <affinity_propagation.h>`
3. Use `clustR::ClustHeader` class methods
4. Add Rcpp dependencies: `// [[Rcpp::depends(ClusterR)]]`

### Available C++ API
- Complete API documented in `inst/include/ClusterRHeader.h`
- Main class: `clustR::ClustHeader` with methods like `mini_batch_kmeans()`, `kmeans_arma()`, etc.
- All R-exposed functions have C++ equivalents for direct linkage

## Testing Strategy
- **Algorithm-specific test files**: Each clustering method has dedicated test file
- **Comprehensive parameter validation**: Tests cover edge cases for all parameters
- **Data compatibility**: Tests use included datasets (`dietary_survey_IBS`, `mushroom`, `soybean`)
- **Cross-platform CI**: Tests run on Windows, macOS, Ubuntu with multiple R versions

## Data Processing Patterns
- **Preprocessing**: `center_scale()` function for standardization before clustering
- **Input validation**: Consistent checks for matrix/data.frame input across all functions
- **Output structure**: All algorithms return lists with standardized components (`clusters`, `centroids`, etc.)
- **S3 class assignment**: Results get appropriate class (`GMMCluster`, `KMeansCluster`, etc.) for method dispatch

## When Making Changes
1. **Algorithm modifications**: Update both R wrapper and C++ implementation
2. **New parameters**: Add validation in R wrapper, update C++ function signature, document in roxygen
3. **Performance changes**: Consider impact on external packages using LinkingTo interface
4. **API changes**: Ensure backward compatibility for LinkingTo consumers