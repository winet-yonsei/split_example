import kfp
import kfp.components as comp
from kfp import dsl
@dsl.pipeline(
    name='ty-kf',
    description='taeyoung kubeflow test'
)


def ty_pipeline():
    image_load = dsl.ContainerOp(
        name="load image data",
        image="hihikim92/ty_kf_load:0.52",
        arguments=[
            '--train', 'false',
            '--out_path', '/image.bin'
        ],
        file_outputs={'image': '/image.bin'}
    )
    layer1 = dsl.ContainerOp(
        name="layer 1",
        image="hihikim92/ty_kf_front:0.51",
        arguments=[
            '--data', image_load.outputs['image'],
            '--train', 'false',
            '--out_path', '/layer1.bin'
        ],
        file_outputs={'layer1': '/layer1.bin'}
    )
    layer2 = dsl.ContainerOp(
        name="layer 2",
        image="hihikim92/ty_kf_middle:0.51",
        arguments=[
            '--data', layer1.outputs['layer1'],
            '--train', 'false',
            '--out_path', '/layer2.bin'
        ],
        file_outputs={'layer2': '/layer2.bin'}
    )
    fc = dsl.ContainerOp(
        name="fc layer",
        image="hihikim92/ty_kf_rear:0.51",
        arguments=[
            '--data', layer2.outputs['layer2'],
            '--train', 'false',
            '--out_data', '/fc_out.bin'
        ],
        file_outputs={'fc_out': '/fc_out.bin'}
    )
    fc.after(layer2)
    layer2.after(layer1)
    layer1.after(image_load)

if __name__=="__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(ty_pipeline, __file__+".tar.gz")