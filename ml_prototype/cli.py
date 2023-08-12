from lightning.pytorch.cli import LightningCLI


def cli_main():
    LightningCLI(save_config_callback=None)


if __name__ == "__main__":
    cli_main()
