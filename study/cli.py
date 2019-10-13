import click
import os
from . import make_dataset, image, video, const, anfis, logger
from .util.path import base_path, dataset_path

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
@click.option('--video')
def dataset(video):
    make_dataset.main(video)


@cli.command("const")
def constant():
    const.main()


@cli.command("anfis")
@click.option('--dataset', default='glcm')
@click.option('--video', default='')
@click.option('--epochs', default=const.EPOCHS)
def anf(dataset, video, epochs):

    dataset_paths = {
        'car'   : dataset_path(os.path.join(video, 'dataset_{0}_for_c.csv'.format(dataset))),
        'truck' : dataset_path(os.path.join(video, 'dataset_{0}_for_t.csv'.format(dataset)))
    }

    click.echo('use {0}'.format(dataset_paths['car']))
    click.echo('use {0}'.format(dataset_paths['truck']))
    click.echo()

    anfis.main(dataset_paths, epochs)
