from app.services.runner_service import RunnerService


def main() -> None:
    runner = RunnerService()
    print("active_collectors=" + ",".join(collector.source_name for collector in runner.collectors))
    results = runner.run_all()
    for source_name, found, saved, status in results:
        print(f"{source_name}: collected={found} saved={saved} status={status}")


if __name__ == "__main__":
    main()
