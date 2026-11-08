/**
 *  @file test/kernel_abi.cpp
 *  @author Ash Vardanian
 *  @date November 8, 2026
 *  @brief The Linux kernel ABI mirrors of @c nvme_device.hpp, checked against the kernel's own
 *      spelling where the build's headers have it.
 */
#include <cstddef> // `offsetof`

#include <linux/io_uring.h>
#include <linux/nvme_ioctl.h>
#include <sys/stat.h> // `struct statx`

#include "usearch/utilities/nvme_device.hpp"

using namespace unum::usearch::utilities;

static_assert(linux_uring_entry_fixed_file_k == IOSQE_FIXED_FILE);
static_assert(linux_nvme_ioctl_id_k == NVME_IOCTL_ID);

static_assert(sizeof(linux_uring_params_t) == sizeof(struct io_uring_params));
static_assert(offsetof(linux_uring_params_t, cq_off) == offsetof(struct io_uring_params, cq_off));
static_assert(sizeof(linux_uring_sqe_t) == sizeof(struct io_uring_sqe));
static_assert(offsetof(linux_uring_sqe_t, user_data) == offsetof(struct io_uring_sqe, user_data));
static_assert(sizeof(linux_uring_cqe_t) == sizeof(struct io_uring_cqe));
static_assert(sizeof(linux_uring_rsrc_update_t) == sizeof(struct io_uring_rsrc_update));
static_assert(sizeof(linux_nvme_passthru_cmd_t) == sizeof(struct nvme_passthru_cmd));

#if defined(IORING_URING_CMD_FIXED) // Linux 6.1 headers also spell the SQE128 command fields
static_assert(offsetof(linux_uring_sqe128_t, cmd_op) == offsetof(struct io_uring_sqe, cmd_op));
static_assert(offsetof(linux_uring_sqe128_t, uring_cmd_flags) == offsetof(struct io_uring_sqe, uring_cmd_flags));
static_assert(offsetof(linux_uring_sqe128_t, cmd) == offsetof(struct io_uring_sqe, cmd));
#endif

#if defined(NVME_URING_CMD_IO)
static_assert(sizeof(linux_nvme_uring_cmd_t) == sizeof(struct nvme_uring_cmd));
#endif

#if defined(STATX_DIOALIGN)
static_assert(offsetof(linux_statx_t, dio_offset_align) == offsetof(struct statx, stx_dio_offset_align));
#endif
