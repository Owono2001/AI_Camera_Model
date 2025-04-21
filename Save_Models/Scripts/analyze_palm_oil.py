# analyze_palm_oil_batch_no_display.py
# Requirements: pytorch, torchvision, pillow (PIL)

import sys
import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with `weights_only=False`.*")

# --- Constants for Formatting ---
SEPARATOR_LINE = "=" * 65
SEPARATOR_SUB = "-" * 65
TITLE_WIDTH = 65

# --- Project Root Determination ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    required_dirs = ["Models", "Captured_Images", "Summary"]
    config_file_path = os.path.join(project_root, "config.py")
    if not all(os.path.isdir(os.path.join(project_root, d)) for d in required_dirs) or not os.path.isfile(config_file_path):
        print(f"⚠️ Warning: Could not reliably determine project root based on expected structure.")
        print(f"   Determined project root: {project_root}")
    sys.path.insert(0, project_root)
    print(f"✅ Project root added to sys.path: {project_root}")
except Exception as e:
    print(f"❌ Error determining project root: {e}")
    sys.exit(1)

# --- Attempt to import project modules ---
try:
    print("Importing project modules...")
    from Models.RGBModel import RGBModel
    from config import args
    print("✅ Successfully imported RGBModel and config.args.")
except ImportError as e:
    print(f"❌ Error importing project modules: {e}")
    sys.exit(1)
except AttributeError as e:
     print(f"❌ Error: 'config.py' does not define 'args' object? {e}")
     sys.exit(1)
except Exception as e:
     print(f"❌ Unexpected error during import: {e}")
     sys.exit(1)

# --- Model Setup ---
def setup_model(expected_config_args):
    """Set up and return the model for inference"""
    print(SEPARATOR_SUB)
    print("🔧 Setting up Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    model_path = os.path.join(project_root, "Summary", "FusionModel", "RGB_Weights", "best_rgb_model.pth")
    # Add alternative path checking if needed...
    if not os.path.exists(model_path):
        # Define potential alternative paths if needed
        alt_model_paths = [
             os.path.join(project_root, "Summary", "FusionModel", "best_rgb_model.pth"),
             os.path.join(project_root, "Save_Models", "Scripts", "best_rgb_model.pth"),
        ]
        found_path = None
        for alt_path in alt_model_paths:
             if os.path.exists(alt_path):
                  found_path = alt_path
                  print(f"   Primary model path not found. Using alternative: {found_path}")
                  break
        if found_path:
             model_path = found_path
        else:
             print(f"   ❌ Model weights not found at primary path: {model_path}")
             print(f"      Also checked alternative paths: {alt_model_paths}")
             raise FileNotFoundError(f"Model weights file 'best_rgb_model.pth' not found.")


    print("   Loading model structure...")
    model = RGBModel(expected_config_args)
    print(f"   Loading model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("   ✅ Model weights loaded successfully (weights_only=True).")
    except Exception as e_wo:
        print(f"   ⚠️ Info: Could not load with weights_only=True ({e_wo}). Trying weights_only=False.")
        try:
             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
             print("   ✅ Model weights loaded successfully (weights_only=False - be cautious).")
        except Exception as e_load:
             print(f"   ❌ Failed to load model weights: {e_load}")
             raise
    model.to(device)
    model.eval()
    print("   ✅ Model ready for inference!")
    print(SEPARATOR_SUB)
    return model, device

# --- Image Loading & Preprocessing ---
def load_image(image_path, img_height, img_width):
    """Load and preprocess a single image for inference."""
    if not isinstance(img_height, int) or not isinstance(img_width, int) or img_height <= 0 or img_width <= 0:
        raise ValueError(f"Invalid image dimensions provided: height={img_height}, width={img_width}")
    try:
        # Keep the original image loading in case it's needed elsewhere later,
        # but we won't use it for display.
        image_pil = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image_pil)
        # Return None for the original image as we don't need it anymore
        return input_tensor, None
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    except Exception as e:
        raise IOError(f"Error loading/transforming image {os.path.basename(image_path)}: {str(e)}")

# --- Image Analysis ---
def analyze_single_image(model, device, image_path, class_info, class_explanations, img_height, img_width, show_image=False):
    """
    Analyzes a single image and returns a dictionary containing results
    and a pre-formatted report string. DOES NOT DISPLAY IMAGE.
    The show_image parameter is now ignored.
    """
    filename = os.path.basename(image_path)
    result_data = {"filename": filename, "status": "error", "message": "Analysis not completed"} # Default error state

    try:
        # Load and preprocess the image
        # We get None for original_image now from load_image
        input_tensor, _ = load_image(image_path, img_height, img_width)

        # Prepare tensor for model
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)
            probabilities = F.softmax(output, dim=1)
            confidences, indices = torch.topk(probabilities, k=len(class_info), dim=1)

        # Process results
        confidences_list = confidences.squeeze().cpu().numpy().tolist()
        indices_list = indices.squeeze().cpu().numpy().tolist()
        if isinstance(confidences_list, float): confidences_list = [confidences_list]
        if isinstance(indices_list, int): indices_list = [indices_list]

        class_keys = list(class_info.keys())

        # --- Map results ---
        detailed_results = []
        invalid_indices = False
        for i in range(len(indices_list)):
            class_index = indices_list[i]
            if 0 <= class_index < len(class_keys):
                class_key = class_keys[class_index]
                if class_key in class_info:
                    class_name, emoji = class_info[class_key]
                    confidence_percent = confidences_list[i] * 100
                    detailed_results.append({
                        "class_key": class_key, "name": class_name,
                        "emoji": emoji, "confidence": confidence_percent
                    })
                else: invalid_indices = True
            else: invalid_indices = True

        if not detailed_results:
             raise ValueError("No valid predictions mapped from model output.")
        if invalid_indices:
             print(f"   ⚠️ Warning: Some prediction indices for {filename} were invalid or unmapped.")

        # Get top prediction details
        top_prediction = detailed_results[0]
        top_class_key = top_prediction["class_key"]
        explanation = class_explanations.get(top_class_key, 'No explanation available.')

        # --- Format the Report String ---
        report_lines = []
        report_lines.append(SEPARATOR_LINE)
        report_lines.append(f"🌴 Analysis: {filename} 🌴".center(TITLE_WIDTH))
        report_lines.append(SEPARATOR_LINE)
        report_lines.append(f"📌 Top Prediction: {top_prediction['emoji']} {top_prediction['name']} ({top_prediction['confidence']:.2f}% confidence)")
        report_lines.append(f"\n🔍 Explanation: {explanation}\n")
        report_lines.append("📊 Confidence Breakdown:")
        for res in detailed_results:
            name_str = res.get('name', 'Unknown')
            report_lines.append(f"   {res.get('emoji','?')} {name_str.ljust(12)}: {res.get('confidence', 0.0):.2f}%")
        report_lines.append(SEPARATOR_LINE)
        report_block = "\n".join(report_lines)
        # --- End Formatting Report String ---

        # Update result data for success
        result_data.update({
            "status": "success",
            "prediction": top_prediction['name'],
            "emoji": top_prediction['emoji'],
            "confidence": top_prediction['confidence'],
            "explanation": explanation,
            "all_confidences": detailed_results,
            "report_block": report_block, # Add the formatted report
            "message": "Analysis successful"
        })

        # --- IMAGE DISPLAY BLOCK REMOVED ---
        # if show_image and original_image:
        #     try:
        #         original_image.show()
        #         result_data["display_status"] = "(Image displayed)" # No longer used
        #     except Exception as e:
        #         print(f"   ⚠️ Could not display image '{filename}': {e}")
        #         result_data["display_status"] = "(Failed to display image)" # No longer used
        # else:
        #      result_data["display_status"] = "" # No longer used

    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        error_message = f"Error analyzing {filename}: {str(e)}"
        print(f"   ❌ {error_message}")
        result_data.update({
            "status": "error",
            "message": error_message,
            "report_block": f"{SEPARATOR_LINE}\n   ❌ Error analyzing {filename}: {e}\n{SEPARATOR_LINE}"
        })
    except Exception as e:
        error_message = f"Unexpected error analyzing {filename}: {str(e)}"
        print(f"   ❌ {error_message}")
        print("      Traceback:")
        traceback.print_exc(limit=2)
        result_data.update({
            "status": "error",
            "message": error_message,
            "report_block": f"{SEPARATOR_LINE}\n   ❌ Unexpected error analyzing {filename}: {e}\n{SEPARATOR_LINE}"
        })

    return result_data


# --- Utility Functions ---
def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
    """Recursively get all image files with specified extensions from a directory."""
    image_files = []
    if not os.path.isdir(directory):
        print(f"❌ Error: Directory not found: {directory}")
        return []
    print(f"🔍 Scanning for images in: {directory} (and subdirectories)")
    count = 0
    try:
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(extensions):
                    image_files.append(os.path.join(root, filename))
                    count += 1
    except Exception as e:
        print(f"❌ Error reading directory {directory}: {e}")
        return []
    print(f"   Found {count} images.")
    return image_files


# --- Reporting ---
def generate_summary_report(results, class_info):
    """Generate a formatted summary report."""
    successful_results = [r for r in results if r and r.get("status") == "success"]
    failed_results = [r for r in results if not r or r.get("status") != "success"]

    total_processed = len(results)
    total_successful = len(successful_results)
    total_failed = len(failed_results)

    summary_lines = []
    summary_lines.append("\n" + SEPARATOR_LINE)
    summary_lines.append("📊🌴 ANALYSIS SUMMARY REPORT 🌴📊".center(TITLE_WIDTH))
    summary_lines.append(SEPARATOR_LINE)
    summary_lines.append(f"Total images processed: {total_processed}")
    summary_lines.append(f"  ✅ Successfully analyzed: {total_successful}")
    summary_lines.append(f"  ❌ Failed analysis:      {total_failed}")

    if total_successful > 0:
        summary_lines.append(SEPARATOR_SUB)
        summary_lines.append("Ripeness Breakdown (Successful Analyses):")
        class_counts = {name: 0 for _, (name, _) in class_info.items()}
        for r in successful_results:
            pred_name = r.get("prediction")
            if pred_name in class_counts: class_counts[pred_name] += 1
            else: class_counts[pred_name] = 1 # Handle unexpected

        for class_name, count in sorted(class_counts.items()):
            percentage = (count / total_successful * 100) if total_successful > 0 else 0
            emoji = "❓"
            for key, (name, emj) in class_info.items():
                if name == class_name: emoji = emj; break
            summary_lines.append(f"  {emoji} {class_name.ljust(12)}: {count} ({percentage:.1f}%)")

        summary_lines.append(SEPARATOR_SUB)
        summary_lines.append("Detailed Results (Successful Analyses):")
        successful_results.sort(key=lambda x: x.get('filename', ''))
        for i, result in enumerate(successful_results, 1):
             summary_lines.append(f"  {i}. {result.get('filename', 'N/A')}: {result.get('emoji', '')} {result.get('prediction', 'Error')} ({result.get('confidence', 0):.1f}%)")
    else:
        summary_lines.append(SEPARATOR_SUB)
        summary_lines.append("No images were successfully analyzed.")

    if total_failed > 0:
         summary_lines.append(SEPARATOR_SUB)
         summary_lines.append("Failed Analyses Details:")
         failed_results.sort(key=lambda x: x.get('filename', ''))
         for i, result in enumerate(failed_results, 1):
              filename = result.get('filename', f'Unknown File {i}')
              message = result.get('message', 'Unknown error')
              message_short = message.split('\n')[0]
              summary_lines.append(f"  {i}. {filename}: {message_short}")

    summary_lines.append(SEPARATOR_LINE)
    print("\n".join(summary_lines))


# --- Main Execution ---
def main():
    # --- Define Class Information ---
    class_info = {
        "empty_bunch": ("Empty Bunch", "🌴❌"),
        "ripe": ("Ripe", "🌴✔️"),
        "unripe": ("Unripe", "🌴🟩")
    }
    try:
         if hasattr(args, 'label_name') and set(args.label_name) != set(class_info.keys()):
              print("⚠️ Warning: Class names mismatch between config.py and analysis script `class_info`.")
              print(f"   config.py: {sorted(args.label_name)}")
              print(f"   script:    {sorted(class_info.keys())}")
    except AttributeError: pass # Ignore if label_name not in args

    class_explanations = {
        "empty_bunch": "Image shows an empty palm oil bunch post-harvest or underdeveloped, no harvestable fruits.",
        "ripe": "Fruits display vibrant orange-red hues, possibly detaching slightly; indicates optimal ripeness for harvest.",
        "unripe": "Predominant green or dark/black color, tightly packed fruits; signifies immature bunch not ready for harvest."
    }

    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='🌴 Palm Oil Fruit Ripeness Analysis Tool 🌴',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('images', nargs='*',
                        help='Optional: Specific image file(s) or directories to analyze.\nPaths can be absolute or relative.')
    default_test_dir = os.path.join(project_root, "Captured_Images", "Test")
    parser.add_argument('--dir', default=default_test_dir,
                        help=f'Target directory for analysis if no paths are given.\nDefault: {default_test_dir}')
    # --no-display argument is kept but now has no effect internally
    parser.add_argument('--no-display', action='store_true',
                        help='Do not attempt to display images (Image display is disabled in this version).')
    default_workers = args.num_workers if hasattr(args, 'num_workers') and isinstance(args.num_workers, int) else None
    parser.add_argument('--max-workers', type=int, default=default_workers,
                        help=f'Max parallel threads. Default from config ({default_workers}) or auto.')
    cli_args = parser.parse_args()

    # --- Print Header ---
    print("\n" + "╔" + "═" * (TITLE_WIDTH - 2) + "╗")
    print(f"║{'🌴 Palm Oil Fruit Ripeness Analysis Tool 🌴'.center(TITLE_WIDTH - 2)}║")
    print("╚" + "═" * (TITLE_WIDTH - 2) + "╝\n")

    # --- Load Model ---
    try:
        model, device = setup_model(args)
    except Exception as e:
        print(f"❌ Critical Error during model setup: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Determine Images to Analyze ---
    images_to_analyze = []
    if cli_args.images:
        print("Processing specific images/directories provided via command line:")
        processed_paths = set()
        for path_arg in cli_args.images:
            path = os.path.abspath(path_arg)
            if path in processed_paths: continue # Skip duplicates

            if os.path.isdir(path):
                 print(f"  -> Scanning directory: {path}")
                 found_files = get_image_files(path)
                 images_to_analyze.extend(f for f in found_files if f not in processed_paths)
                 processed_paths.update(found_files)
            elif os.path.isfile(path):
                 if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                     print(f"  -> Adding image file: {path}")
                     images_to_analyze.append(path)
                     processed_paths.add(path)
                 else:
                     print(f"  -> Skipping non-image file: {path}")
                     processed_paths.add(path) # Add to prevent re-checking if listed again
            else:
                 print(f"❌ Warning: Specified path not found or not valid: {path_arg}")
                 processed_paths.add(path) # Add to prevent re-checking
    else:
        target_dir = os.path.abspath(cli_args.dir)
        print(f"No specific paths provided. Processing target directory set by --dir.")
        if not os.path.isdir(target_dir):
             print(f"❌ Error: Target directory '{target_dir}' not found.")
             sys.exit(1)
        images_to_analyze = get_image_files(target_dir)


    if not images_to_analyze:
        print("❌ No valid images found to analyze. Exiting.")
        sys.exit(0)

    # Ensure uniqueness and sort
    images_to_analyze = sorted(list(set(images_to_analyze)))
    total_images = len(images_to_analyze)
    print(f"✅ Found {total_images} unique images to analyze.")

    # --- Retrieve and Validate Image Dimensions ---
    try:
         img_h = args.img_height
         img_w = args.img_width
         if not isinstance(img_h, int) or not isinstance(img_w, int) or img_h <=0 or img_w <=0:
              raise ValueError(f"Invalid dimensions in config (must be positive integers)")
         print(f"   Using image dimensions from config: {img_w}W x {img_h}H")
    except (AttributeError, ValueError) as e:
         print(f"❌ CRITICAL Error retrieving image dimensions from config.py: {e}")
         print("   Ensure --img_height and --img_width are defined correctly in config.py.")
         sys.exit(1)
    print(SEPARATOR_SUB)

    # --- Parallel Analysis Execution ---
    print(f"🚀 Starting analysis of {total_images} images...")
    start_time = time.time()
    results_list = []

    num_workers = cli_args.max_workers
    worker_info = f"{num_workers} workers" if num_workers else "default number of workers"
    print(f"   Using {worker_info}.")
    print(SEPARATOR_SUB)

    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='AnalyzeWorker') as executor:
        future_to_image = {
            executor.submit(analyze_single_image, model, device, img_path, class_info,
                            class_explanations, img_h, img_w, False): # show_image=False permanently
            img_path for img_path in images_to_analyze
        }

        processed_count = 0
        for future in as_completed(future_to_image):
            processed_count += 1
            image_path = future_to_image[future]
            filename = os.path.basename(image_path)

            print(f"[{processed_count}/{total_images}] Analyzing: {filename}...")

            try:
                result_data = future.result()
                results_list.append(result_data)
                print(result_data.get("report_block", "Error: Report block missing.")) # Print report

                if result_data.get("status") == "success":
                    # Removed display_msg from here
                    print(f"[{processed_count}/{total_images}] ✅ Done: {filename}")
                else:
                    print(f"[{processed_count}/{total_images}] ❌ Error processing: {filename}")

            except Exception as exc:
                print(f"\n❌ Exception occurred retrieving result for {filename}: {exc}")
                traceback.print_exc(limit=2)
                results_list.append({"filename": filename, "status": "error",
                                     "message": f"Result retrieval exception: {exc}",
                                     "report_block": f"{SEPARATOR_LINE}\n   ❌ Error retrieving result for {filename}: {exc}\n{SEPARATOR_LINE}"})
                print(f"[{processed_count}/{total_images}] ❌ Error processing: {filename}")
            print() # Blank line for spacing

    # --- Final Summary and Timing ---
    elapsed_time = time.time() - start_time
    print(SEPARATOR_LINE)
    print("🏁 Analysis Complete 🏁".center(TITLE_WIDTH))
    print(SEPARATOR_LINE)

    generate_summary_report(results_list, class_info)

    successful_count = sum(1 for r in results_list if r and r.get("status") == "success")
    avg_time = elapsed_time / total_images if total_images > 0 else 0

    print(f"⏱️ Total analysis time: {elapsed_time:.2f} seconds ({avg_time:.2f} seconds/image avg)")
    print(f"   Successfully analyzed: {successful_count} / {total_images} images")
    print("\n" + "═" * TITLE_WIDTH + "\n")

if __name__ == "__main__":
    main()