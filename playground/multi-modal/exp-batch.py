from fire import Fire
from pymemorial import nsys, send_slack


def main():
    X_batch_size = [1, 2, 4, 8, 16]
    for batch_size in X_batch_size:
        nsys.profile(
            cuda_visible_devices="4",
            gpu_metrics_device="2",
            output=f"./reports/prof-b{batch_size}.nsys-rep",
            stats=True,
            target_program=f"python run-vllm.py --modality image --num-prompts {batch_size}",
        )
        nsys.stats(
            report="nvtx_gpu_proj_sum",
            output_prefix=f"./csv/prof-b{batch_size}",
            target_sqlite=f"./reports/prof-b{batch_size}.sqlite",
        )
        send_slack(f"Finished profiling batch size {batch_size}")


if __name__ == "__main__":
    Fire(main)
