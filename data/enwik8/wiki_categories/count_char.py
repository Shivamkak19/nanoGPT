import os
from pathlib import Path


def count_characters_in_directory(directory_path):
    """
    Count total characters across all text files in the given directory.

    Args:
        directory_path (str): Path to the directory to analyze

    Returns:
        tuple: (total_characters, files_processed, errors)
    """
    total_chars = 0
    files_processed = 0
    errors = []

    # Convert to Path object for better cross-platform compatibility
    dir_path = Path(directory_path)

    # Common text file extensions
    text_extensions = {".txt"}

    try:
        for file_path in dir_path.iterdir():
            # Skip if not a file or doesn't have text extension
            if (
                not file_path.is_file()
                or file_path.suffix.lower() not in text_extensions
            ):
                continue

            try:
                # Read file with UTF-8 encoding
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    total_chars += len(content)
                    files_processed += 1

            except UnicodeDecodeError:
                errors.append(
                    f"Unable to read {file_path}: Not a valid UTF-8 text file"
                )
            except Exception as e:
                errors.append(f"Error processing {file_path}: {str(e)}")

    except Exception as e:
        errors.append(f"Error accessing directory: {str(e)}")

    return total_chars, files_processed, errors


# Example usage
if __name__ == "__main__":
    directory = "subclasses"  # Current directory, change this to your target directory

    total_chars, files_processed, errors = count_characters_in_directory(directory)

    print(f"\nTotal characters: {total_chars:,}")
    print(f"Files processed: {files_processed}")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"- {error}")
