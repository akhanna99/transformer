def process_input(input_data):
    # Implement your Python script logic here to process the input_data
    result = f"Processed input: {input_data}"
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: your_script.py <input_data>")
    else:
        input_data = sys.argv[1]
        processed_result = process_input(input_data)
        print(processed_result)
