import os
from faker import Faker

def generate_names_for_countries(country_names, num=2000):
    for country in country_names:
        fake = Faker(country)
        names = [fake.name() for _ in range(num)]
        file_path = f"../data/name/{country}.txt"
        with open(file_path, 'w') as file:
            for name in names:
                file.write(name + '\n')
        print(f"{num} names generated for {country}. Saved to {file_path}")

# List of country names
# country_names = ['ja_JP', 'ru_RU', 'ta_IN', 'hi_IN', 'fr_CA', 'en', 'es_MX', 'en_IE', 'el_CY', 'la']
country_names = ['fr_CA', 'en', 'es_MX', 'en_IE', 'el_CY', 'la']

# Generate names for each country and save to .txt files
generate_names_for_countries(country_names)
