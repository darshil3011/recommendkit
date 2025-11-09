#!/usr/bin/env python3
"""
Correlated Dataset Generator for Recommendation System Testing
Creates a dataset with PERFECT correlations between user attributes and item interactions
"""

import json
import random
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Set seed for reproducibility
random.seed(42)

class CorrelatedDatasetGenerator:
    def __init__(self, num_users: int = 1000, num_items: int = 100):
        self.num_users = num_users
        self.num_items = num_items
        
        # Define occupations and their target categories
        self.occupations = {
            'software_engineer': {'categories': ['tech'], 'base_salary': 80000, 'count': 100},
            'doctor': {'categories': ['medical_equipment'], 'base_salary': 120000, 'count': 100},
            'chef': {'categories': ['kitchen'], 'base_salary': 45000, 'count': 100},
            'mechanical_engineer': {'categories': ['tools'], 'base_salary': 75000, 'count': 100},
            'teacher': {'categories': ['educational'], 'base_salary': 50000, 'count': 100},
            'artist': {'categories': ['art'], 'base_salary': 40000, 'count': 100},
            'lawyer': {'categories': ['business'], 'base_salary': 110000, 'count': 100},
            'driver': {'categories': ['automotive'], 'base_salary': 42000, 'count': 100},
            'nurse': {'categories': ['healthcare'], 'base_salary': 65000, 'count': 100},
            'scientist': {'categories': ['lab'], 'base_salary': 85000, 'count': 100}
        }
        
        # Define locations and their weather items
        self.locations = {
            'USA': {'weather_items': ['cold'], 'count': 200},
            'India': {'weather_items': ['hot'], 'count': 200},
            'Canada': {'weather_items': ['cold', 'rain'], 'count': 200},
            'Australia': {'weather_items': ['hot'], 'count': 200},
            'UK': {'weather_items': ['rain'], 'count': 200}
        }
        
        # Define age groups and their preferences
        self.age_groups = {
            'young': {'age_range': (18, 30), 'items': ['smartphone', 'tablet', 'smartwatch']},
            'middle': {'age_range': (31, 45), 'items': ['office_chair', 'desk_lamp']},
            'senior': {'age_range': (46, 60), 'items': ['prescription_glasses', 'heating_pad']},
            'elderly': {'age_range': (61, 75), 'items': ['walking_cane', 'wheelchair']}
        }
        
        # Define item catalog by category
        self.item_categories = {
            'tech': [
                'Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Webcam', 'Headphones',
                'Smartphone', 'Tablet', 'Smartwatch', 'USB Drive', 'External Hard Drive', 'Phone Charger'
            ],
            'medical_equipment': [
                'Stethoscope', 'Thermometer', 'Blood Pressure Monitor', 'Pulse Oximeter', 
                'Medical Gloves', 'Surgical Mask', 'First Aid Kit', 'Wheelchair', 
                'Walking Cane', 'Prescription Glasses'
            ],
            'kitchen': [
                'Chef Knife Set', 'Cutting Board', 'Cookware Set', 'Blender', 
                'Microwave', 'Toaster', 'Coffee Maker', 'Food Processor', 
                'Mixing Bowls', 'Apron'
            ],
            'tools': [
                'Drill Machine', 'Wrench Set', 'Screwdriver Set', 'Hammer', 
                'Measuring Tape', 'Safety Helmet', 'Work Gloves', 'Toolbox', 
                'Level', 'Pliers'
            ],
            'educational': [
                'Whiteboard', 'Markers', 'Textbooks', 'Notebooks', 
                'Pen Set', 'Calculator', 'Backpack', 'Globe'
            ],
            'art': [
                'Paint Set', 'Paint Brushes', 'Canvas', 'Sketchbook',
                'Easel', 'Color Palette', 'Drawing Pencils', 'Art Portfolio'
            ],
            'business': [
                'Business Suit', 'Briefcase', 'Leather Shoes', 'Tie',
                'Document Organizer', 'File Cabinet', 'Desk Lamp', 'Office Chair'
            ],
            'automotive': [
                'Car Vacuum', 'Dash Cam', 'GPS Navigator', 'Car Phone Mount',
                'Emergency Kit', 'Tire Pressure Gauge', 'Car Charger', 'Seat Covers'
            ],
            'healthcare': [
                'Hand Sanitizer', 'Face Masks', 'Medical Scrubs', 'Compression Socks',
                'Heating Pad', 'Ice Pack', 'Bandages', 'Medical Scissors'
            ],
            'lab': [
                'Lab Coat', 'Safety Goggles', 'Test Tubes', 
                'Microscope', 'Lab Gloves', 'Petri Dishes'
            ],
            'weather_cold': [
                'Winter Jacket', 'Heater', 'Thermal Wear', 'Gloves'
            ],
            'weather_hot': [
                'Portable Fan', 'Cooler', 'Sunglasses', 'Sunscreen'
            ],
            'weather_rain': [
                'Umbrella', 'Raincoat', 'Rain Boots', 'Waterproof Bag'
            ]
        }
        
        self.items = []
        self.users = []
        self.item_id_map = {}  # Maps (category, item_name) to item_id
        
    def generate_dataset(self) -> Dict[str, Any]:
        """Generate the complete correlated dataset"""
        print(f"🎯 Generating correlated dataset with {self.num_users} users and ~{self.num_items} items...")
        
        # Generate items
        self.items = self._generate_items()
        print(f"✅ Generated {len(self.items)} items")
        
        # Generate users with correlations
        self.users = self._generate_users()
        print(f"✅ Generated {len(self.users)} users")
        
        # Generate interactions based on perfect correlations
        interactions = self._generate_correlated_interactions()
        print(f"✅ Generated {len(interactions)} interactions")
        
        return {
            'user_data': self.users,
            'item_data': self.items,
            'interactions': interactions
        }
    
    def _generate_items(self) -> List[Dict]:
        """Generate all items with their categories"""
        items = []
        item_id = 1
        
        for category, item_names in self.item_categories.items():
            for item_name in item_names:
                # Determine price based on category
                if category == 'tech':
                    price = random.uniform(200, 2000)
                elif category in ['medical_equipment', 'lab']:
                    price = random.uniform(50, 500)
                elif category in ['kitchen', 'tools']:
                    price = random.uniform(30, 300)
                elif category == 'business':
                    price = random.uniform(100, 800)
                elif category in ['art', 'educational']:
                    price = random.uniform(10, 150)
                else:
                    price = random.uniform(20, 200)
                
                item = {
                    'item_id': item_id,
                    'categorical': {
                        'category': category
                    },
                    'continuous': {
                        'price': round(price, 2)
                    },
                    'text': {
                        'name': item_name,
                        'description': f"High-quality {item_name} for professionals and enthusiasts"
                    }
                }
                items.append(item)
                self.item_id_map[(category, item_name)] = item_id
                item_id += 1
        
        return items
    
    def _generate_users(self) -> List[Dict]:
        """Generate users with balanced distribution"""
        users = []
        user_id = 1
        
        # Create distribution lists
        occupation_list = []
        for occupation, info in self.occupations.items():
            occupation_list.extend([occupation] * info['count'])
        
        location_list = []
        for location, info in self.locations.items():
            location_list.extend([location] * info['count'])
        
        # Shuffle to randomize assignment
        random.shuffle(occupation_list)
        random.shuffle(location_list)
        
        for i in range(self.num_users):
            occupation = occupation_list[i]
            location = location_list[i]
            
            # Determine age group
            age_group = random.choice(list(self.age_groups.keys()))
            age_range = self.age_groups[age_group]['age_range']
            age = random.randint(age_range[0], age_range[1])
            
            # Determine salary based on occupation and age
            base_salary = self.occupations[occupation]['base_salary']
            age_factor = 1.0 + (age - 25) * 0.02  # Older = higher salary
            salary = base_salary * age_factor * random.uniform(0.9, 1.3)
            
            # Generate interaction history
            previous_items = self._generate_user_interactions(occupation, location, age_group, salary)
            
            user = {
                'user_id': user_id,
                'categorical': {
                    'occupation': occupation,
                    'location': location
                },
                'continuous': {
                    'age': float(age),
                    'salary': round(salary, 2)
                },
                'temporal': {
                    'previous_liked_items': previous_items
                }
            }
            users.append(user)
            user_id += 1
        
        return users
    
    def _generate_user_interactions(self, occupation: str, location: str, 
                                    age_group: str, salary: float) -> List[int]:
        """Generate interaction history based on user attributes"""
        item_ids = []
        
        # 1. PRIMARY: Occupation-based items (100% correlation)
        occupation_categories = self.occupations[occupation]['categories']
        for category in occupation_categories:
            for item in self.items:
                if item['categorical']['category'] == category:
                    item_ids.append(item['item_id'])
        
        # 2. SECONDARY: Location-based weather items (100% correlation)
        weather_types = self.locations[location]['weather_items']
        for weather_type in weather_types:
            weather_category = f'weather_{weather_type}'
            for item in self.items:
                if item['categorical']['category'] == weather_category:
                    item_ids.append(item['item_id'])
        
        # 3. TERTIARY: Age-based items (DISABLED - causes cross-contamination)
        # age_items = self.age_groups[age_group]['items']
        # for item_name in age_items:
        #     # Find item by name (special age items)
        #     for item in self.items:
        #         if item['text']['name'].lower().replace(' ', '_') == item_name.lower():
        #             item_ids.append(item['item_id'])
        
        # 4. QUATERNARY: Salary-based premium items (DISABLED - breaks clean correlations)
        # if salary > 80000:
        #     # High earners also buy premium tech items
        #     for item in self.items:
        #         if item['categorical']['category'] == 'tech' and item['continuous']['price'] > 1000:
        #             if item['item_id'] not in item_ids:
        #                 item_ids.append(item['item_id'])
        
        return sorted(list(set(item_ids)))  # Remove duplicates and sort
    
    def _generate_correlated_interactions(self) -> List[Dict]:
        """Generate ONLY positive interactions - data loader will handle negatives"""
        interactions = []
        
        for user in self.users:
            user_id = user['user_id']
            liked_items = user['temporal']['previous_liked_items']
            
            # Create positive interactions for ALL correlated items
            for item_id in liked_items:
                interaction = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'timestamp': self._random_timestamp()
                }
                interactions.append(interaction)
        
        return interactions
    
    def _random_timestamp(self) -> str:
        """Generate a random timestamp within the last year"""
        days_ago = random.randint(1, 365)
        timestamp = datetime.now() - timedelta(days=days_ago)
        return timestamp.isoformat()
    
    def print_statistics(self, dataset: Dict[str, Any]):
        """Print comprehensive dataset statistics"""
        print(f"\n{'='*60}")
        print(f"📊 DATASET STATISTICS")
        print(f"{'='*60}")
        
        print(f"\n🔢 Counts:")
        print(f"   Users: {len(dataset['user_data'])}")
        print(f"   Items: {len(dataset['item_data'])}")
        print(f"   Interactions: {len(dataset['interactions'])}")
        
        # Occupation distribution
        print(f"\n👔 Occupation Distribution:")
        occupation_counts = {}
        for user in dataset['user_data']:
            occ = user['categorical']['occupation']
            occupation_counts[occ] = occupation_counts.get(occ, 0) + 1
        for occ, count in sorted(occupation_counts.items()):
            print(f"   {occ}: {count} users")
        
        # Location distribution
        print(f"\n🌍 Location Distribution:")
        location_counts = {}
        for user in dataset['user_data']:
            loc = user['categorical']['location']
            location_counts[loc] = location_counts.get(loc, 0) + 1
        for loc, count in sorted(location_counts.items()):
            print(f"   {loc}: {count} users")
        
        # Category distribution
        print(f"\n📦 Item Category Distribution:")
        category_counts = {}
        for item in dataset['item_data']:
            cat = item['categorical']['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        for cat, count in sorted(category_counts.items()):
            print(f"   {cat}: {count} items")
        
        # Interaction statistics
        print(f"\n💬 Interaction Statistics:")
        print(f"   Total interactions: {len(dataset['interactions'])}")
        print(f"   All positive (negatives will be sampled by data loader)")
        print(f"   Avg interactions per user: {len(dataset['interactions']) / len(dataset['user_data']):.1f}")
        
        # Sample user analysis
        print(f"\n👤 Sample User Analysis:")
        for i in [0, 250, 500]:
            user = dataset['user_data'][i]
            print(f"\n   User {user['user_id']}:")
            print(f"      Occupation: {user['categorical']['occupation']}")
            print(f"      Location: {user['categorical']['location']}")
            print(f"      Age: {user['continuous']['age']:.0f}")
            print(f"      Salary: ${user['continuous']['salary']:,.0f}")
            print(f"      Interacted items: {len(user['temporal']['previous_liked_items'])}")
            
            # Show item categories
            item_categories = set()
            for item_id in user['temporal']['previous_liked_items']:
                item = next(it for it in dataset['item_data'] if it['item_id'] == item_id)
                item_categories.add(item['categorical']['category'])
            print(f"      Item categories: {', '.join(sorted(item_categories))}")
        
        print(f"\n{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate correlated synthetic dataset')
    parser.add_argument('--num_users', type=int, default=1000, 
                       help='Number of users (default: 1000)')
    parser.add_argument('--num_items', type=int, default=100, 
                       help='Target number of items (default: 100)')
    parser.add_argument('--output', type=str, default='correlated_dataset.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = CorrelatedDatasetGenerator(args.num_users, args.num_items)
    dataset = generator.generate_dataset()
    
    # Save to file
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Print statistics
    generator.print_statistics(dataset)
    
    print(f"\n✅ Dataset generation complete!")


if __name__ == "__main__":
    main()

