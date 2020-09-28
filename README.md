# Heart Disease Data

## How to run locally

1. Clone the project and switch into the directory.
```sh
git clone https://github.com/YukiHinana/heartdata.git 
cd heartdata
```
2. Setup a pipenv shell and install the requirements.
We assume you have pipenv and python3.
```sh
# create the virtual environment
pipenv shell
# Install the requirements
pipenv install
```
3. Run the program within your virtual environment.
```sh
python heartdata.py
```
4. It should show you that it is listening on localhost:5000.

5. Open [http://localhost:5000](http://localhost:5000) in your browser.

6. Import an appropriately formatted CSV file and select whether or not the first column is a variable or an index.
   * Some properly formatted data sets are provided in the github as an example, and can be run through the program as well.

7. Refresh the page as many times as desired to rerun the machine learning model on a random set of your data.
