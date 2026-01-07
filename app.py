# import base64
# from io import BytesIO

# from flask import Flask, render_template, request
# import matplotlib.pyplot as plt
# import my_tf_mod  # your own helper

# app = Flask(__name__)

# # ========================
# # Nutrition info per 100g
# # ========================
# nutrition_data = {
#     "Apple": {
#         "Calories": "52 kcal",
#         "Carbs": "14 g",
#         "Protein": "0.3 g",
#         "Fat": "0.2 g",
#         "Fiber": "2.4 g",
#         "Vitamin C": "7% DV",
#     },
#     "Banana": {
#         "Calories": "96 kcal",
#         "Carbs": "27 g",
#         "Protein": "1.3 g",
#         "Fat": "0.3 g",
#         "Fiber": "2.6 g",
#         "Vitamin C": "14% DV",
#     },
#     "Orange": {
#         "Calories": "47 kcal",
#         "Carbs": "12 g",
#         "Protein": "0.9 g",
#         "Fat": "0.1 g",
#         "Fiber": "2.4 g",
#         "Vitamin C": "89% DV",
#     },
# }

# # map common model outputs -> keys in nutrition_data
# name_map = {
#     "apple": "Apple",
#     "banana": "Banana",
#     "orange": "Orange",
# }

# def normalize_label(label: str) -> str:
#     """Normalize model label to our nutrition keys."""
#     if not label:
#         return "Unknown"
#     key = label.strip().lower()
#     return name_map.get(key, label.strip().title())  # fall back to Title Case


# @app.route('/')
# def home():
#     return render_template('home.html')


# @app.route('/Prediction', methods=['GET', 'POST'])
# def pred():
#     # Default (GET or no file): empty page with no results
#     if request.method != 'POST' or 'file' not in request.files or not request.files['file'].filename:
#         return render_template(
#             'Pred3.html',
#             fruit_dict=None,
#             rotten=None,
#             plot_url=None,
#             fruit_name=None,
#             nutrition_info=None
#         )

#     # ---- POST with file ----
#     file = request.files['file']

#     # Your helper: returns original image (array) and preprocessed input
#     org_img, img = my_tf_mod.preprocess(file)

#     # Classifiers
#     fruit_dict = my_tf_mod.classify_fruit(img)  # expected dict like {'apple': 92.3, 'banana': 3.1, 'orange': 4.6}
#     rotten = my_tf_mod.check_rotten(img)        # e.g. [fresh%, rotten%]

#     # Pick top fruit label robustly
#     fruit_name_raw = None
#     if isinstance(fruit_dict, dict) and fruit_dict:
#         fruit_name_raw = max(fruit_dict, key=fruit_dict.get)
#     elif isinstance(fruit_dict, (list, tuple)) and fruit_dict:
#         fruit_name_raw = str(fruit_dict[0])

#     fruit_name = normalize_label(fruit_name_raw or "Unknown")
#     nutrition_info = nutrition_data.get(fruit_name, {})

#     # --- Inline image as base64 ---
#     img_x = BytesIO()
#     plt.imshow(org_img / 255.0)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(img_x, format='png', bbox_inches='tight', pad_inches=0)
#     plt.close()
#     img_x.seek(0)
#     plot_url = base64.b64encode(img_x.getvalue()).decode('utf8')

#     # (Optional) debug prints in console to verify keys/case
#     print("fruit_dict =", fruit_dict)
#     print("top raw =", fruit_name_raw, "| normalized =", fruit_name)
#     print("nutrition keys available:", list(nutrition_data.keys()))

#     return render_template(
#         'Pred3.html',
#         fruit_dict=fruit_dict,
#         rotten=rotten,
#         plot_url=plot_url,
#         fruit_name=fruit_name,
#         nutrition_info=nutrition_info
#     )


# if __name__ == '__main__':
#     app.run(debug=True)

# NEXT CODE

import base64
from io import BytesIO

from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend
import matplotlib.pyplot as plt

import my_tf_mod  # your own helper

app = Flask(__name__)

# ========================
# Nutrition info per 100g
# ========================
nutrition_data = {
    "Apple": {
        "Calories": "52 kcal",
        "Carbs": "14 g",
        "Protein": "0.3 g",
        "Fat": "0.2 g",
        "Fiber": "2.4 g",
        "Vitamin C": "7% DV",
    },
    "Banana": {
        "Calories": "96 kcal",
        "Carbs": "27 g",
        "Protein": "1.3 g",
        "Fat": "0.3 g",
        "Fiber": "2.6 g",
        "Vitamin C": "14% DV",
    },
    "Orange": {
        "Calories": "47 kcal",
        "Carbs": "12 g",
        "Protein": "0.9 g",
        "Fat": "0.1 g",
        "Fiber": "2.4 g",
        "Vitamin C": "89% DV",
    },
}

# map common model outputs -> keys in nutrition_data
name_map = {
    "apple": "Apple",
    "banana": "Banana",
    "orange": "Orange",
}

def normalize_label(label: str) -> str:
    """Normalize model label to our nutrition keys."""
    if not label:
        return "Unknown"
    key = label.strip().lower()
    return name_map.get(key, label.strip().title())  # fallback to Title Case


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/Prediction', methods=['GET', 'POST'])
def pred():
    if request.method != 'POST' or 'file' not in request.files or not request.files['file'].filename:
        return render_template(
            'Pred3.html',
            fruit_dict=None,
            rotten=None,
            plot_url=None,
            fruit_name=None,
            nutrition_info=None
        )

    file = request.files['file']

    # Get original image and preprocessed input
    org_img, img = my_tf_mod.preprocess(file)

    # Get predictions
    fruit_dict = my_tf_mod.classify_fruit(img)
    rotten = my_tf_mod.check_rotten(img)  # Expected: [fresh%, rotten%]

    # Get most probable fruit
    fruit_name_raw = None
    if isinstance(fruit_dict, dict) and fruit_dict:
        fruit_name_raw = max(fruit_dict, key=fruit_dict.get)
    elif isinstance(fruit_dict, (list, tuple)) and fruit_dict:
        fruit_name_raw = str(fruit_dict[0])

    fruit_name = normalize_label(fruit_name_raw or "Unknown")

    # Only show nutrition info if the fruit is fresh
    fresh_percentage = rotten[0] if isinstance(rotten, (list, tuple)) and len(rotten) >= 2 else 0
    is_fresh = fresh_percentage >= 50  # You can change threshold
    nutrition_info = nutrition_data.get(fruit_name, {}) if is_fresh else None

    # Generate inline image
    img_x = BytesIO()
    plt.imshow(org_img / 255.0)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_x, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    img_x.seek(0)
    plot_url = base64.b64encode(img_x.getvalue()).decode('utf8')

    # Debug logs
    print("fruit_dict =", fruit_dict)
    print("top raw =", fruit_name_raw, "| normalized =", fruit_name)
    print("fresh % =", fresh_percentage, "| is fresh =", is_fresh)
    print("nutrition keys available:", list(nutrition_data.keys()))

    return render_template(
        'Pred3.html',
        fruit_dict=fruit_dict,
        rotten=rotten,
        plot_url=plot_url,
        fruit_name=fruit_name,
        nutrition_info=nutrition_info
    )


if __name__ == '__main__':
    app.run(debug=True)



####

# import base64
# from io import BytesIO

# from flask import Flask, render_template, request
# import matplotlib
# matplotlib.use('Agg')  # ✅ Use non-GUI backend
# import matplotlib.pyplot as plt

# import my_tf_mod  # your own helper

# app = Flask(__name__)

# # ========================
# # Nutrition info per 100g
# # ========================
# nutrition_data = {
#     "Apple": {
#         "Calories": "52 kcal",
#         "Carbs": "14 g",
#         "Protein": "0.3 g",
#         "Fat": "0.2 g",
#         "Fiber": "2.4 g",
#         "Vitamin C": "7% DV",
#     },
#     "Banana": {
#         "Calories": "96 kcal",
#         "Carbs": "27 g",
#         "Protein": "1.3 g",
#         "Fat": "0.3 g",
#         "Fiber": "2.6 g",
#         "Vitamin C": "14% DV",
#     },
#     "Orange": {
#         "Calories": "47 kcal",
#         "Carbs": "12 g",
#         "Protein": "0.9 g",
#         "Fat": "0.1 g",
#         "Fiber": "2.4 g",
#         "Vitamin C": "89% DV",
#     },
# }

# # map common model outputs -> keys in nutrition_data
# name_map = {
#     "apple": "Apple",
#     "banana": "Banana",
#     "orange": "Orange",
# }

# def normalize_label(label: str) -> str:
#     """Normalize model label to our nutrition keys."""
#     if not label:
#         return "Unknown"
#     key = label.strip().lower()
#     return name_map.get(key, label.strip().title())  # fallback to Title Case


# @app.route('/')
# def home():
#     return render_template('home.html')


# @app.route('/Prediction', methods=['GET', 'POST'])
# def pred():
#     if request.method != 'POST' or 'file' not in request.files or not request.files['file'].filename:
#         return render_template(
#             'Pred3.html',
#             fruit_dict=None,
#             rotten=None,
#             plot_url=None,
#             fruit_name=None,
#             nutrition_info=None
#         )

#     file = request.files['file']

#     # Get original image and preprocessed input
#     org_img, img = my_tf_mod.preprocess(file)

#     # Get predictions
#     fruit_dict = my_tf_mod.classify_fruit(img)
#     rotten = my_tf_mod.check_rotten(img)  # Expected: [fresh%, rotten%]

#     # Get most probable fruit
#     fruit_name_raw = None
#     if isinstance(fruit_dict, dict) and fruit_dict:
#         fruit_name_raw = max(fruit_dict, key=fruit_dict.get)
#     elif isinstance(fruit_dict, (list, tuple)) and fruit_dict:
#         fruit_name_raw = str(fruit_dict[0])

#     fruit_name = normalize_label(fruit_name_raw or "Unknown")

#     # Only show nutrition info if the fruit is fresh
#     fresh_percentage = rotten[0] if isinstance(rotten, (list, tuple)) and len(rotten) >= 2 else 0
#     is_fresh = fresh_percentage >= 50  # You can change threshold
#     nutrition_info = nutrition_data.get(fruit_name, {}) if is_fresh else None

#     # Generate inline image (✅ FIXED: no /255.0 here)
#     img_x = BytesIO()
#     plt.imshow(org_img.astype("uint8"))  # display properly
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(img_x, format='png', bbox_inches='tight', pad_inches=0)
#     plt.close()
#     img_x.seek(0)
#     plot_url = base64.b64encode(img_x.getvalue()).decode('utf8')

#     # Debug logs
#     print("fruit_dict =", fruit_dict)
#     print("top raw =", fruit_name_raw, "| normalized =", fruit_name)
#     print("fresh % =", fresh_percentage, "| is fresh =", is_fresh)
#     print("nutrition keys available:", list(nutrition_data.keys()))

#     return render_template(
#         'Pred3.html',
#         fruit_dict=fruit_dict,
#         rotten=rotten,
#         plot_url=plot_url,
#         fruit_name=fruit_name,
#         nutrition_info=nutrition_info
#     )


# if __name__ == '__main__':
#     app.run(debug=True)
