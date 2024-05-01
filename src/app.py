from flask import Flask, render_template, request, jsonify
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load the language model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/console', methods=['POST'])
def console():
    command = request.form['command']
    response = execute_command(command)
    return jsonify({'response': response})

def execute_command(command):
    if command.startswith('python'):
        # Extract the command and arguments
        command_parts = command.split(' ', 1)
        command = command_parts[0]
        args = command_parts[1] if len(command_parts) > 1 else ''

        # Generate a response using the language model
        input_ids = tokenizer.encode(command + args, return_tensors='pt')
        output = model.generate(input_ids, max_length=1024, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text
    else:
        # Execute the command in a real Linux command line interface
        try:
            output = os.popen(command).read()
            return output
        except Exception as e:
            return f"Command '{command}' not found: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)