import cv2 as cv
import numpy as np
import potrace
from potrace import Bitmap
import matplotlib.pyplot as plt

def plot_curve(ax, curve, latex_list):
    # Get the starting point of the curve
    start_point = curve.start_point
    
    # Extract points for plotting the curve
    curve_points = []

    for segment in curve.segments:
        if isinstance(segment, potrace.BezierSegment):
            # Extract control points and end point
            c1 = segment.c1
            c2 = segment.c2
            end_point = segment.end_point

            # Invert y-coordinates for plotting the curve
            inverted_points = [[x, -y] for x, y in [start_point, c1, c2, end_point]]
            curve_points.append(inverted_points)

            # Update the start_point for the next segment
            start_point = end_point

    # Plot the curve
    for points in curve_points:
        bezier_curve = np.array(points).reshape(-1, 2)
        ax.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'b-')

        # Append equation for the current Bezier curve to the latex list
        equation = f"((1-t)((1-t)((1-t){points[0][0]}+t{points[1][0]})+t((1-t){points[1][0]}+t{points[2][0]}))+t((1-t)((1-t){points[1][0]}+t{points[2][0]})+t((1-t){points[2][0]}+t{points[3][0]})), (1-t)((1-t)((1-t){points[0][1]}+t{points[1][1]})+t((1-t){points[1][1]}+t{points[2][1]}))+t((1-t)((1-t){points[1][1]}+t{points[2][1]})+t((1-t){points[2][1]}+t{points[3][1]})))"
        latex_list.append(equation)

def plot_path(ax, path, latex_list):
    # Plot each curve in the path
    for curve in path.curves:
        plot_curve(ax, curve, latex_list)

def plot_edges_and_curves(image_path, output_file):
    # Load the image
    frame = cv.imread(image_path)

    # Check if the image is loaded successfully
    if frame is None:
        print("Error: Could not open or read the image.")
        exit()

    # Convert the image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv.Canny(gray, 150, 150)

    # Convert Canny edges to a binary representation (0 or 255)
    edges_binary = np.where(edges > 0, 1, 0).astype(np.uint8)

    # Create a Potrace Bitmap
    bitmap = Bitmap(edges_binary)

    # Trace the bitmap
    latex_list = []  # List to store equations
    path = bitmap.trace(turdsize=50, turnpolicy=potrace.TURNPOLICY_MINORITY, alphamax=1.3, opticurve=1, opttolerance=0.03)

    # Create a Matplotlib figure and axis for Bezier curves
    fig1, ax1 = plt.subplots()

    # Plot Bezier curves for edges
    plot_path(ax1, path, latex_list)

    # Write equations to the output file
    with open(output_file, 'w') as file:
        for equation in latex_list:
            file.write(equation + '\n')

    # Create a Matplotlib figure and axis for original Canny edges
    fig2, ax2 = plt.subplots()

    # Display the original Canny edges
    ax2.imshow(edges, cmap='gray')

    # Show the plots
    plt.show()

# Example usage
image_path = '/home/torazo/Downloads/face grapher 3/autodraw 19_11_2023.png'
output_file = 'equations.txt'
plot_edges_and_curves(image_path, output_file)
