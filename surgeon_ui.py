import streamlit as st
from io import StringIO

def generate_prescription_content(doctor_name, patient_name, prioritized_lenses, lens_brands):
    content = f"Doctor Name: {doctor_name}\n"
    
    # Sort lenses by rank and extract only the lens names
    sorted_lenses = [lens for rank, lens in sorted(prioritized_lenses.items())]
    content += f"Prioritized Lenses: {', '.join(sorted_lenses)}\n"
    
    # Collect and join brands for prioritized lenses
    prioritized_brands = [lens_brands[lens] for lens in sorted_lenses if lens_brands.get(lens)]
    content += f"Brands: {', '.join(prioritized_brands)}\n"
    
    content += f"Patient Name: {patient_name}\n"
    
    return content

def main():
    st.set_page_config(page_title="Surgeon Interface for IOL Selection", layout="wide")
    
    st.title("Surgeon Interface for IOL Selection")
    doctor_name = st.text_input("Doctor Name")
    patient_name = st.text_input("Patient Name")
    
    st.subheader("Prioritized Lenses")
    lens_types = ["Monofocal", "Multifocal", "Toric", "Light Adjustable"]
    prioritized_lenses = {1: "Monofocal"}  # Monofocal is fixed at rank 1
    lens_brands = {}
    
    st.write("Monofocal (Rank 1 - Fixed)")
    
    for lens in lens_types[1:]:  # Skip Monofocal as it's already ranked
        col1, col2 = st.columns([1, 1])
        with col1:
            include = st.checkbox(f"Include {lens}", key=f"include_{lens}")
        with col2:
            if include:
                rank = st.text_input(f"Rank for {lens} (2-4)", key=f"rank_{lens}")
                if rank:
                    try:
                        rank = int(rank)
                        if 2 <= rank <= 4:
                            prioritized_lenses[rank] = lens
                        else:
                            st.error(f"Invalid rank for {lens}. Please enter a number between 2 and 4.")
                    except ValueError:
                        st.error(f"Invalid input for {lens} rank. Please enter a number.")

    st.subheader("Brand Preference")
    
    for lens in lens_types:
        lens_brands[lens] = st.text_input(f"{lens} Brand", key=f"brand_{lens}")
    
    if st.button("Generate Prescription"):
        if not doctor_name or not patient_name or len(prioritized_lenses) == 1:
            st.error("Please fill in the Doctor Name, Patient Name, and select at least one additional lens type.")
        elif len(prioritized_lenses) != len(set(prioritized_lenses.values())):
            st.error("Each selected lens must have a unique rank. Please adjust the ranks.")
        else:
            prescription_content = generate_prescription_content(doctor_name, patient_name, prioritized_lenses, lens_brands)
            st.success("Prescription generated successfully!")
            
            filename = f"{patient_name.replace(' ', '_')}_IOL_prescription.txt"
            
            st.download_button(
                label="Download Prescription",
                data=prescription_content,
                file_name=filename,
                mime="text/plain"
            )

if __name__ == "__main__":
    main()