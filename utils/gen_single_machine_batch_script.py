from sys import argv


def main():
    num_total, batch_idx, total_batch = argv[1:]
    num_total = int(num_total)
    batch_idx = int(batch_idx)
    total_batch = int(total_batch)

    # assert total_batch == 4

    result = [
        # this way id is always consistent with order in nvidia-smi
        # see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
        # https://stackoverflow.com/questions/26123252/inconsistency-of-ids-between-nvidia-smi-l-and-cudevicegetname
        # https://codeyarns.com/2016/07/05/how-to-make-cuda-and-nvidia-smi-use-same-gpu-id/
        'export CUDA_DEVICE_ORDER=PCI_BUS_ID',
        f'export CUDA_VISIBLE_DEVICES={batch_idx}',
    ]
    for idx in range(batch_idx, num_total, total_batch):
        # sleep is to give some chance for ctrl+c
        result.append(f'echo {idx}.sh\n./{idx}.sh &> /dev/null\nsleep 1')

    print('\n'.join(result))


if __name__ == '__main__':
    main()
