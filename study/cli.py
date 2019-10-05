import click
from . import make_dataset, image, video, const, anfis, logger


@click.group("cli")
@click.option("--log", default="INFO", help="Logging level of the logger.")
def cli(log):
    logger.set_level(log)


@cli.command("image")
def img():
    image.main()


@cli.command("video")
def vid():
    video.main()


@cli.command("make_dataset")
def dataset():
    make_dataset.main()


@cli.command("const")
def constant():
    const.main()


@cli.command()
def anfis():
    anfis.main()
