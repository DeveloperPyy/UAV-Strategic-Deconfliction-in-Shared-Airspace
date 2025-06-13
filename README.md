# UAV Strategic Deconfliction in Shared Airspace

## Objective
This project implements a strategic deconfliction system for Unmanned Aerial Vehicles (UAVs) to ensure safe flight operations in shared airspace. It acts as a final authority to verify whether a drone's planned waypoint mission can be safely executed by checking for spatial and temporal conflicts against simulated flight paths of other drones. The system supports 4D (3D spatial coordinates + time) simulation for enhanced conflict detection.

## Features
* **4D Spatio-Temporal Conflict Detection:** Identifies conflicts in 3D space over time, considering a defined safety buffer.
* **Detailed Conflict Explanation:** Provides specific locations, times, and involved drones for any detected conflicts.
* **Flexible Scenario Management:** Loads drone missions and simulated flight schedules from a configurable JSON file.
* **Comprehensive Visualization:**
    * Animated 3D trajectories (Matplotlib GIF and interactive Plotly HTML).
    * Distance vs. Time plots for primary vs. simulated drones.
    * Temporal Conflict Timeline to highlight conflict durations.
* **Automated Reporting:** Generates a text-based deconfliction report for each scenario.
* **Modular and Extensible Design:** Code is organized into distinct modules for easy maintenance and future expansion.

## Getting Started

### Prerequisites
* Python 3.8+ (Python 3.12.10 was used for this project)
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DeveloperPyy/UAV-Strategic-Deconfliction-in-Shared-Airspace.git
    cd UAV-Strategic-Deconfliction-in-Shared-Airspace
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file if you don't have one. You can generate it by running `pip freeze > requirements.txt` after installing all dependencies.)*


[//]: # (4.  **Install external dependencies for Matplotlib GIF animations &#40;if on Windows&#41;:**)

[//]: # (    * Matplotlib's GIF animations often rely on `ImageMagick`.)

[//]: # (    * Download and install `ImageMagick` from [https://imagemagick.org/]&#40;https://imagemagick.org/&#41;.)

[//]: # (    * During installation, ensure you select the options to **"Install legacy utilities &#40;e.g. convert&#41;"** and **"Add application directory to your system PATH."**)

### Usage

1.  **Configure Scenarios:**
    Edit the `data/simulated_flights.json` file to define your drone missions, simulated drone paths, and global settings (safety buffer, time step). Example scenarios are already provided in the file.

2.  **Run the Simulation:**
    From the project's root directory, execute the `main.py` script:
    ```bash
    python src/main.py
    ```

3.  **View Outputs:**
    After execution, all generated reports, plots, and animations will be saved in the `media/` directory:
    * **`media/reports/`**: Text reports detailing conflict status.
    * **`media/animations/`**: Matplotlib GIF animations for each scenario.
    * **`media/animations/plotly_animations/`**: Interactive Plotly HTML animations for each scenario (open these in a web browser).
    * **`media/plots/`**: PNG images of Distance vs. Time plots and Temporal Conflict Timelines.

## Project Structure


## AI Assistance Acknowledgment
This project was developed with assistance from an AI language model (Google's Gemini). The AI aided in:
* Brainstorming architectural approaches and design patterns.
* Debugging complex errors and providing solutions.
* Explaining concepts and providing code examples.
* Ensuring adherence to best practices and common Python conventions.

## Contact
For questions or feedback, reach out to **Shashank**, GitHub profile: [DeveloperPyy](https://github.com/DeveloperPyy).


## License
MIT License

Copyright (c) 2025 Shashank Dharpure

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.