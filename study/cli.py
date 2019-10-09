import click
from . import make_dataset, image, video, const, anfis, logger
from .util.path import get_base_path

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


@cli.command("anfis")
@click.option('--dataset', '-d', default='glcm')
def anf(dataset):

    dataset_paths = {
        'car'   : get_base_path('dataset/dataset_{0}_for_c.csv'.format(dataset)),
        'truck' : get_base_path('dataset/dataset_{0}_for_t.csv'.format(dataset))
    }

    click.echo('use {0}'.format(dataset_paths['car']))
    click.echo('use {0}'.format(dataset_paths['truck']))

    anfis.main(dataset_paths)
