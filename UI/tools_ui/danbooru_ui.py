



def show_danbooru_area():
    global image_in
    with gr.Row():
        with gr.Column(variant="compact"):
            image_in = gr.Image(label="input image", type="pil", elem_id="image_init")
    with gr.Row():
        apply_btn = gr.Button("Analyze image with Deep DanBooru", variant="primary")
        mem_btn = gr.Button("Unload from memory")
    with gr.Row():
        results = gr.Textbox(value="", lines=8, label="Results")

    mem_btn.click(fn=unload_DanBooru, inputs=results , outputs=results)
    apply_btn.click(fn=analyze_DanBooru, inputs=image_in , outputs=results)