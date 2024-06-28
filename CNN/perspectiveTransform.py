import cv2
import numpy as np

# Turn on Laptop's webcam
cap = cv2.VideoCapture(0)

# Define 10 different perspective transformations
perspectives = [
    np.float32([[0, 0], [640, 0], [0, 480], [640, 480]]),  # Original
    np.float32([[0, 480], [640, 480], [0, 0], [640, 0]]),  # Vertical flip
    np.float32([[640, 0], [0, 0], [640, 480], [0, 480]]),  # Horizontal flip
    np.float32([[100, 100], [540, 100], [0, 480], [640, 480]]),  # Trapezoid (top smaller)
    np.float32([[0, 0], [640, 140], [0, 480], [640, 340]]),  # Skew right
    np.float32([[0, 140], [640, 0], [0, 340], [640, 480]]),  # Skew left
    np.float32([[100, 0], [540, 0], [0, 480], [640, 480]]),  # Trapezoid (top narrower)
    np.float32([[0, 0], [640, 0], [100, 480], [540, 480]]),  # Trapezoid (bottom narrower)
    np.float32([[0, 0], [320, 0], [0, 480], [320, 480]]),  # Left half
    np.float32([[320, 0], [640, 0], [320, 480], [640, 480]])  # Right half
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = []

    # Source points (assuming 640x480 webcam resolution, adjust if different)
    pts1 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])

    # Apply each perspective transform
    for i, pts2 in enumerate(perspectives):
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (640, 480))
        # Add perspective number to the image
        cv2.putText(result, f"Perspective {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        results.append(result)

    # Create a 2x5 grid to display all 10 images
    grid = np.zeros((960, 3200, 3), dtype=np.uint8)
    for i, img in enumerate(results):
        row = i // 5
        col = i % 5
        grid[row*480:(row+1)*480, col*640:(col+1)*640] = img

    # Resize the grid to fit on most screens
    display_grid = cv2.resize(grid, (1600, 480))

    # Display the grid
    cv2.imshow('Multiple Perspectives', display_grid)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()