"""Apply a small, reproducible management hook to the pinned official UI."""
import ast


def patch_character_ui(repo,work):
    source=repo/'GPT_SoVITS/inference_webui_fast.py'
    text=source.read_text(encoding='utf-8')
    if '# CHARACTER_MANAGEMENT_V1' in text: return
    original=text
    signature='def inference(\n    text,'
    if signature not in text: raise ValueError('Official UI signature changed; management patch needs review.')
    text=text.replace(signature,'def inference(\n    character_id,\n    task_id,\n    text,',1)
    text=text.replace('for item in tts_pipeline.run(inputs):','for item in character_panel.run(character_id, task_id, inputs):',1)
    block='with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:\n'
    if block not in text: raise ValueError('Official Blocks layout changed.')
    text=text.replace(block,block+'    # CHARACTER_MANAGEMENT_V1\n    from character_webui import CharacterPanel\n'
                      '    character_panel = CharacterPanel(tts_pipeline)\n    character_panel.header()\n',1)
    text=text.replace('            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])\n','',1)
    start=text.index('        SoVITS_dropdown.change(\n')
    end_marker='        GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])'
    end=text.index(end_marker,start)+len(end_marker)
    text=text[:start]+('        character_panel.bind(GPT_dropdown, SoVITS_dropdown, inp_ref, prompt_text,\n'
                       '                             prompt_language, inference_button, seed, inp_refs, refresh_button)')+text[end:]
    click='        inference_button.click(\n            inference,\n            [\n'
    if click not in text: raise ValueError('Official inference button changed.')
    text=text.replace(click,click+'                character_panel.character,\n                character_panel.task,\n',1)
    text=text.replace('value=os.environ.get("local_ref_audio")','value=character_panel.reference_audio',1)
    text=text.replace('value=os.environ.get("local_ref_text", "")','value=character_panel.reference_text',1)
    text=text.replace('value=i18n("日文") if os.environ.get("local_ref_audio") else i18n("中文")',
                      'value=character_panel.reference_language',1)
    ast.parse(text)
    backup=work/'upstream_originals/GPT_SoVITS/inference_webui_fast_before_characters.py'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists(): backup.write_text(original,encoding='utf-8')
    source.write_text(text,encoding='utf-8')
