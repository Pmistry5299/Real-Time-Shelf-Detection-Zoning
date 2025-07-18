🛒 Real-Time Shelf Detection & Zoning (Demo)
This is a demo project built with Python and OpenCV to detect shelves in a video stream and divide them into logical product zones — just like how retail stores segment shelves by brand or category.

⚠️ Note: This is a proof-of-concept on simulated shelves, not trained or tested on real retail data (yet).

💡 What It Does
📷 Captures live video from an IP camera or webcam

🔍 Detects shelf-like rectangles using contours and angle filtering

➖ Identifies horizontal lines to verify multi-level shelves

🎨 Divides each detected shelf into zones (3–4–3–4 pattern)

🧠 Tracks shelves across frames using IOU-based grouping

✅ Overlays color-coded segments for easy visualization

🧰 Tech Stack
Python

OpenCV

NumPy

IP Camera/Webcam Stream

Basic CV techniques: Canny, HoughLines, Contours, Bounding Boxes

📦 Future Ideas (Open for Contributions)
🏷️ Product detection/classification within zones

📊 Integration with planogram or inventory databases

📷 Real-world retail shelf training data

📡 Barcode/QR code detection per zone

📱 AR overlay for mobile or smart glasses use cases

🏃‍♂️ How to Run
Clone the repo

Install dependencies: pip install -r requirements.txt

Update IP camera URL in the script (or fallback to webcam)

Run: python shelf_detection.py

Press q to quit

🤝 Contributions & Feedback
This is an experimental project built just for fun and learning.
Feel free to fork, play around, and suggest improvements!

Let’s make shelves smarter — even in demos. 😉
