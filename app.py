# import os
# os.system('python setup.py develop')

import json
from pathlib import Path

import gradio as gr
import matplotlib
import random

from gradio_utils.utils import (process_img, process_query_img, get_select_coords, get_select_bbox_coords, select_skeleton,
                                reset_skeleton, reset_kp, reset_bbox, process, update_examples, toggle_coarse_fine, select_model,
                                save_support_query, confirm_kp)

import os

random.seed(0)

LENGTH = 480  # Length of the square area displaying/editing images

matplotlib.use('agg')
model_dir = Path('./checkpoints')
TIMEOUT = 80

with gr.Blocks() as demo:
    gr.Markdown('''
    # We introduce EdgeCape, a novel framework that overcomes these limitations by predicting the graph's edge weights which optimizes localization. 
    To further leverage structural priors, we propose integrating Markovian Structural Bias, which modulates the self-attention interaction between nodes based on the number of hops between them. 
    We show that this improves the model’s ability to capture global spatial dependencies. 
    Evaluated on the MP-100 benchmark, which includes 100 categories and over 20K images, 
    EdgeCape achieves state-of-the-art results in the 1-shot setting and leads among similar-sized methods in the 5-shot setting, significantly improving keypoint localization accuracy.
    ### [Paper](https://arxiv.org/pdf/2411.16665) | [Project Page](https://orhir.github.io/edge_cape/) 
    ## Instructions
    1. Upload an image of the object you want to pose.
    2. Mark keypoints on the image.
    3. Mark limbs on the image.
    4. Upload an image of the object you want to pose to the query image (**bottom**).
    5. Click **Evaluate** to pose the query image.
    ''')

    models_list = gr.State({
        'EdgeCape-Split1': {
            'config': 'configs/test/1shot_split1.py',
            'checkpoint': 'Checkpoints/1shot_split1.pth',
        },
        'EdgeCape-Split2': {
            'config': 'configs/test/1shot_split2.py',
            'checkpoint': 'Checkpoints/1shot_split2.pth',
        },
        'EdgeCape-Split3': {
            'config': 'configs/test/1shot_split3.py',
            'checkpoint': 'Checkpoints/1shot_split3.pth',
        },
        'EdgeCape-Split4': {
            'config': 'configs/test/1shot_split4.py',
            'checkpoint': 'Checkpoints/1shot_split4.pth',
        },
        'EdgeCape-Split5': {
            'config': 'configs/test/1shot_split5.py',
            'checkpoint': 'Checkpoints/1shot_split5.pth',
        },
        'GraphCape-Split1': {
            'config': 'configs/test/pam/1shot_split1.py',
            'checkpoint': 'Checkpoints/pam/1shots_graph_split1.pth',
        },
        'GraphCape-Split4': {
            'config': 'configs/test/pam/1shot_split4.py',
            'checkpoint': 'Checkpoints/pam/1shots_graph_split4.pth',
        },
        'GraphCape-Split3': {
            'config': 'configs/test/pam/1shot_split3.py',
            'checkpoint': 'Checkpoints/pam/1shots_graph_split3.pth',
        },
        'GraphCape-Split5': {
            'config': 'configs/test/pam/1shot_split5.py',
            'checkpoint': 'Checkpoints/pam/1shots_graph_split5.pth',
        },
        'CapeFormer-Split1': {
            'config': 'configs/test/capeformer/1shot_split1.py',
            'checkpoint': 'Checkpoints/capeformer/capeformer-split1-1shot-4c40dfd2_20230713.pth',
        },
        'CapeFormer-Split4': {
            'config': 'configs/test/capeformer/1shot_split4.py',
            'checkpoint': 'Checkpoints/capeformer/capeformer-split4-1shot-1a3f6b2e_20230713.pth',
        },
        'EdgeCapeRefined-Split1': {
            'config': 'configs/test/refined_1shot_split1_human_face.py',
            'checkpoint': 'Checkpoints/1shot_split1.pth',
        },
        'EdgeCapeRefined-Split4': {
            'config': 'configs/test/refined_1shot_split4_human_face.py',
            'checkpoint': 'Checkpoints/1shot_split4.pth',
        },
        'GraphCapeRefined-Split1': {
            'config': 'configs/test/pam/refined_1shot_split1_human_face.py',
            'checkpoint': 'Checkpoints/pam/1shots_graph_split1.pth',
        },
        'GraphCapeRefined-Split4': {
            'config': 'configs/test/pam/refined_1shot_split4_human_face.py',
            'checkpoint': 'Checkpoints/pam/1shots_graph_split4.pth',
        },
        'CapeFormerRefined-Split1': {
            'config': 'configs/test/capeformer/refined_1shot_split1_human_face.py',
            'checkpoint': 'Checkpoints/capeformer/capeformer-split1-1shot-4c40dfd2_20230713.pth',
        },
        'CapeFormerRefined-Split4': {
            'config': 'configs/test/capeformer/refined_1shot_split4_human_face.py',
            'checkpoint': 'Checkpoints/capeformer/capeformer-split4-1shot-1a3f6b2e_20230713.pth',
        },
    })

    global_state = gr.State({
        "bbox": [],
        "images": {},
        "points": [],
        "coarse": [],
        "skeleton": [],
        "prev_point": None,
        "curr_type_point": "start",
        "load_example": False,
        "is_coarse": True,
    })

    tmp_folder = Path("tmp/gradio/saved")
    if tmp_folder.exists():
        with open(tmp_folder / "saved_state.jsonl", "r") as f:
            saved_data = [json.loads(line) for line in f.readlines()]
    else:
        saved_data = []

    print(f"Loaded {len(saved_data)} saved examples")

    examples_list = gr.State([
        ['examples/dog2.png',
            'examples/dog1.png',
            json.dumps({
                'points': [(232, 200), (312, 204), (228, 264), (316, 472), (316, 616), (296, 868), (412, 872),
                        (416, 624), (604, 608), (648, 860), (764, 852), (696, 608), (684, 432)],
                'coarse': [1.0] * 13,
                'skeleton': [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5),
                            (3, 7), (7, 6), (3, 12), (12, 8), (8, 9),
                            (12, 11), (11, 10)],
            })
            ],
        ['examples/sofa1.jpg',
            'examples/sofa2.png',
            json.dumps({'points': [[272, 561], [193, 482], [339, 460], [445, 530], [264, 369], [203, 318], [354, 300],
                                [457, 341], [345, 63], [187, 68]],
                        'coarse': [1.0] * 10,
                        'skeleton': [[0, 4], [1, 5], [2, 6], [3, 7], [7, 6], [6, 5],
                                    [5, 4], [4, 7], [5, 9], [9, 8], [8, 6]],
            })],
        ['examples/person1.jpeg',
            'examples/person2.jpeg',
            json.dumps({
                'points': [[322, 488], [431, 486], [526, 644], [593, 486], [697, 492], [407, 728],
                        [522, 726], [625, 737], [515, 798]],
                'coarse': [1.0] * 9,
                'skeleton': [[0, 1], [1, 3], [3, 4], [1, 2], [2, 3], [5, 6], [6, 7], [7, 8], [8, 5]],
            })] 
    ] + [
            [str(item['support_image']),
                str(item['query_image']),
                json.dumps({
                    'points': item['points'],
                    'coarse': item['coarse'],
                    'skeleton': item['skeleton'],
                })
                ] for item in saved_data
    ])


    with gr.Row():
        # Upload & Preprocess Image Column
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Upload & Preprocess Image</p>"""
            )
            support_image = gr.Image(
                height=LENGTH,
                width=LENGTH,
                type="pil",
                image_mode="RGB",
                label="Support Image",
                show_label=True,
                interactive=True,
            )

        # Click Points Column
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Click Points</p>"""
            )
            kp_support_image = gr.Image(
                type="pil",
                label="Keypoints Image",
                show_label=True,
                height=LENGTH,
                width=LENGTH,
                interactive=False,
                show_fullscreen_button=False,
            )
            with gr.Row():
                coarse_fine_button = gr.Checkbox(label="Use coarse", value=True, scale=3)
            with gr.Row():
                confirm_kp_button = gr.Button("Confirm Clicked Points", scale=3)
            with gr.Row():
                undo_kp_button = gr.Button("Undo Clicked Points", scale=3)

        # Editing Results Column
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Click Skeleton</p>"""
            )
            skel_support_image = gr.Image(
                type="pil",
                label="Skeleton Image",
                show_label=True,
                height=LENGTH,
                width=LENGTH,
                interactive=False,
                show_fullscreen_button=False,
            )
            with gr.Row():
                pass
            with gr.Row():
                undo_skel_button = gr.Button("Undo Skeleton")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Query Image</p>"""
            )
            query_image = gr.Image(
                type="pil",
                image_mode="RGB",
                label="Query Image",
                show_label=True,
                interactive=True,
            )
            with gr.Row():
                undo_bbox_button = gr.Button("Undo Bounding Box")
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Output</p>"""
            )
            # output_img = gr.Plot(label="Output Image")
            output_img = gr.Image(
                type="pil",
                label="Output Image",
                show_label=True,
                interactive=False,
            )
            additional_plot = gr.Plot(label="Additional Visualization", )
    with gr.Row():
        save_button = gr.Button("Save Support and Query Images")
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=list(models_list.value.keys()),
                                     label="Select Model",
                                     interactive=True,
                                     )
    with gr.Row():
        eval_btn = gr.Button(value="Evaluate")
    with gr.Row():
        gr.Markdown("## Examples")
    with gr.Row():
        example_null = gr.Textbox(type='text',
                                  visible=False
                                  )
    with gr.Row():
        examples = gr.Examples(examples_list.value,
            inputs=[support_image, query_image, example_null],
            outputs=[support_image, kp_support_image, skel_support_image, query_image, global_state],
            fn=update_examples,
            run_on_click=True,
            examples_per_page=5,
        )

    support_image.upload(process_img,
                         inputs=[support_image, global_state],
                         outputs=[kp_support_image, skel_support_image, global_state])
    query_image.upload(process_query_img,
                       inputs=[query_image, global_state],
                       outputs=[global_state])
    kp_support_image.select(get_select_coords,
                            [coarse_fine_button, global_state],
                            [global_state, kp_support_image],
                            queue=False, )
    query_image.select(get_select_bbox_coords,
                       [query_image, global_state],
                       [global_state, query_image],
                       queue=False, )
    coarse_fine_button.change(fn=toggle_coarse_fine,
                              inputs=[coarse_fine_button],
                              outputs=[])
    confirm_kp_button.click(confirm_kp,
                            inputs=global_state,
                            outputs=[skel_support_image, global_state])
    undo_kp_button.click(reset_kp,
                         inputs=global_state,
                         outputs=[kp_support_image, skel_support_image, global_state])
    undo_skel_button.click(reset_skeleton,
                           inputs=global_state,
                           outputs=[skel_support_image, global_state])
    undo_bbox_button.click(fn=reset_bbox,
                           inputs=global_state,
                           outputs=[query_image, global_state])
    skel_support_image.select(select_skeleton,
                              inputs=[global_state],
                              outputs=[global_state, skel_support_image])
    model_dropdown.change(fn=select_model,
                            inputs=[model_dropdown, global_state],
                            outputs=[])
    eval_btn.click(fn=process,
                   inputs=[query_image, global_state, models_list, model_dropdown],
                   outputs=[output_img, additional_plot],)
    save_button.click(fn=save_support_query,
                      inputs=[examples_list, global_state],
                      outputs=[examples.dataset, examples_list])

if __name__ == "__main__":
    print("Start app")
    gr.close_all()
    demo.launch(show_api=False)
