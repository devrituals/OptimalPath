from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.list import MDList, OneLineListItem
import requests
import json

class LocationInput(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = '10dp'
        self.padding = '10dp'
        
        self.lat_input = MDTextField(
            hint_text="Latitude",
            helper_text="Enter latitude",
            helper_text_mode="on_error",
        )
        self.lon_input = MDTextField(
            hint_text="Longitude",
            helper_text="Enter longitude",
            helper_text_mode="on_error",
        )
        self.name_input = MDTextField(
            hint_text="Location Name (Optional)",
            helper_text="Enter location name",
            helper_text_mode="on_error",
        )
        
        self.add_widget(self.lat_input)
        self.add_widget(self.lon_input)
        self.add_widget(self.name_input)

class MainScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.locations = []
        
        # Main layout
        layout = MDBoxLayout(orientation='vertical', padding='10dp', spacing='10dp')
        
        # Title
        title = MDLabel(
            text="Optimal Path Finder",
            halign="center",
            size_hint_y=None,
            height='48dp',
            font_style="H5"
        )
        layout.add_widget(title)
        
        # Location inputs
        self.location_inputs = LocationInput()
        layout.add_widget(self.location_inputs)
        
        # Add location button
        add_button = MDRaisedButton(
            text="Add Location",
            on_release=self.add_location
        )
        layout.add_widget(add_button)
        
        # Locations list
        self.locations_list = MDList()
        scroll = MDScrollView()
        scroll.add_widget(self.locations_list)
        layout.add_widget(scroll)
        
        # Calculate path button
        calculate_button = MDRaisedButton(
            text="Calculate Optimal Path",
            on_release=self.calculate_path
        )
        layout.add_widget(calculate_button)
        
        # Results
        self.results_label = MDLabel(
            text="",
            halign="center",
            size_hint_y=None,
            height='100dp'
        )
        layout.add_widget(self.results_label)
        
        self.add_widget(layout)
    
    def add_location(self, *args):
        try:
            lat = float(self.location_inputs.lat_input.text)
            lon = float(self.location_inputs.lon_input.text)
            name = self.location_inputs.name_input.text or f"Location {len(self.locations) + 1}"
            
            self.locations.append({
                "latitude": lat,
                "longitude": lon,
                "name": name
            })
            
            self.locations_list.add_widget(
                OneLineListItem(text=f"{name}: ({lat}, {lon})")
            )
            
            # Clear inputs
            self.location_inputs.lat_input.text = ""
            self.location_inputs.lon_input.text = ""
            self.location_inputs.name_input.text = ""
            
        except ValueError:
            self.location_inputs.lat_input.error = True
            self.location_inputs.lon_input.error = True
    
    def calculate_path(self, *args):
        if len(self.locations) < 2:
            self.results_label.text = "Please add at least 2 locations"
            return
        
        try:
            response = requests.post(
                "http://localhost:8000/calculate-path",
                json={"locations": self.locations}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.results_label.text = (
                    f"Path calculated successfully!\n"
                    f"Total distance: {result['total_distance']:.2f} km\n"
                    f"Estimated time: {result['estimated_time']:.2f} minutes"
                )
            else:
                self.results_label.text = f"Error: {response.text}"
                
        except Exception as e:
            self.results_label.text = f"Error: {str(e)}"

class OptimalPathApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Light"
        return MainScreen()

if __name__ == "__main__":
    OptimalPathApp().run() 