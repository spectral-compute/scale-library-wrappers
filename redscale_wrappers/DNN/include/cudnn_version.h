/* Note: some libraries (e.g. PyTorch) regex-match the file contents to determine the version. */

#define CUDNN_MAJOR 9
#define CUDNN_MINOR 0
#define CUDNN_PATCHLEVEL 0

// This definition is provided by the documentation.
#define CUDNN_VERSION (CUDNN_MAJOR * 10000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
