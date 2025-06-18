import json
import os
import argparse
from jinja2 import Environment, FileSystemLoader


def read_build_template(path):
    try:
        with open(path, 'r') as f:
            content = f.read()
            return content if content.endswith('\n') else content + '\n'
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
    image_model_init = read_build_template(f'playground/architectures/{image_arch}/build.template')
    text_model_init = read_build_template(f'playground/architectures/{text_arch}/build.template')

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
        evaluators=evaluators
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