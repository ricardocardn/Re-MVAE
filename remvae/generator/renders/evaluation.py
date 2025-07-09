import json
import os
import argparse
from jinja2 import Environment, FileSystemLoader


def read_build_template(path: str):
    try:
        with open(path, 'r') as f:
            content = f.read()
            return content if content.endswith('\n') else content + '\n'
    except:
        return ""


def indent_block(text: str, base_indent: int = 4):
    lines = text.splitlines()
    if not lines:
        return ''
    indent = ' ' * base_indent
    
    indented_lines = [indent + line for line in lines]
    return '\n'.join(indented_lines) + '\n'


def read_and_indent_call(path: str, indent_spaces: int = 4):
    try:
        with open(path, 'r') as f:
            content = f.read()
        single_line = ' '.join(content.split())
        indent = ' ' * indent_spaces
        return indent + single_line + '\n\n'
    except:
        return ""


def generate_eval_script(args_path: str):
    with open(args_path, 'r') as f:
        args = json.load(f)

    reader = args['reader']
    image_arch = args['image_architecture']
    text_arch = args['text_architecture']
    evaluators = args.get('evaluators', [])

    libs = read_build_template(f'playground/architectures/{text_arch}/libs.template')
    image_model_init_raw = read_build_template(f'playground/architectures/{image_arch}/build.template')
    text_model_init_raw = read_build_template(f'playground/architectures/{text_arch}/build.template')

    image_model_init = indent_block(image_model_init_raw)
    text_model_init = indent_block(text_model_init_raw)

    evaluator_imports = ', '.join(evaluators)

    evaluator_calls = ""
    for evaluator in evaluators:
        path = f'playground/evaluators/{evaluator}/{evaluator}.call.template'
        evaluator_calls += read_and_indent_call(path, indent_spaces=4)

    evaluator_functions = ""
    for evaluator in evaluators:
        path = f'playground/evaluators/{evaluator}/{evaluator}.template'
        evaluator_functions += read_build_template(path) + '\n'

    env = Environment(
        loader=FileSystemLoader('.'),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True
    )

    template = env.get_template('generator/templates/eval.py.jinja')
    rendered_code = template.render(
        reader=reader,
        image_architecture=image_arch,
        text_architecture=text_arch,
        libs=libs,
        image_model_init=image_model_init,
        text_model_init=text_model_init,
        evaluator_imports=evaluator_imports,
        evaluator_calls=evaluator_calls,
        evaluator_functions=evaluator_functions
    )

    output_dir = f'experiments/{args["name"]}'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'eval.py')
    with open(output_path, 'w') as f:
        f.write(rendered_code)

    args_output_path = os.path.join(output_dir, 'args.json')
    with open(args_output_path, 'w') as f:
        json.dump(args, f, indent=4)

    print(f"Evaluation script generated at: {output_path}")
    print(f"Arguments saved at: {args_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation script from template and args.json")
    parser.add_argument("args_path", type=str, help="Path to args.json")
    args = parser.parse_args()
    generate_eval_script(args.args_path)


if __name__ == "__main__":
    main()