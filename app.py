import os
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Load API Key ---
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found in .env file.")
    print("Please create a .env file with: GROQ_API_KEY=your_key_here")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

def generate_with_retry(prompt, temperature=0.85, max_tokens=600, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a world-renowned creative director and art critic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait_time = (2 ** attempt) * 5
                print(f"⏳ Rate limit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded.")

def main():
    print("\n" + "=" * 70)
    print("🌟 DREAM FORGE PRO")
    print("you direct · AI creates · together you make masterpieces")
    print("=" * 70 + "\n")
    
    # Get inputs
    shape = input("🎯 Enter a shape (e.g., circle, star, dragon): ").strip() or "circle"
    style = input("🎨 Enter art style (e.g., Surrealist, Cyberpunk): ").strip() or "Surrealist"
    mood = input("🌊 Enter mood (e.g., Dreamy, Mysterious): ").strip() or "Dreamy"
    colors = input("🎨 Enter color palette (e.g., Vibrant, Warm): ").strip() or "Vibrant"
    feedback = input("✏️ Enter your creative feedback: ").strip() or "Make it more colorful"
    
    print("\n" + "-" * 70)
    print("🎨 Forging your masterpiece...\n")
    
    try:
        # Step 1: Ideas
        print("💡 Generating creative concepts...")
        ideas_prompt = f"""
        User wants artwork based on {shape} in {style} style with {mood} mood and {colors} colors.
        Generate 3 creative concepts. Include title, description, emotional impact, and techniques.
        User feedback: {feedback}
        """
        ideas = generate_with_retry(ideas_prompt)
        print(f"\n💡 VISIONS:\n{ideas}\n")
        
        # Step 2: Critic
        print("🔍 Analyzing and critiquing...")
        critic_prompt = f"""
        Evaluate these ideas: {ideas}
        Pick the best one and suggest one improvement.
        """
        critic = generate_with_retry(critic_prompt, temperature=0.7)
        print(f"\n🔍 REFLECTION:\n{critic}\n")
        
        # Step 3: Final
        print("✨ Polishing final masterpiece...")
        final_prompt = f"""
        Refine the best idea: {critic}
        Create a beautiful 2-sentence description.
        """
        final = generate_with_retry(final_prompt, temperature=0.9)
        print(f"\n✨ MASTERWORK:\n{final}\n")
        
        # Save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"masterpieces/masterwork_{timestamp}.txt"
        os.makedirs("masterpieces", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"🌟 MASTERWORK: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Shape: {shape}\nStyle: {style}\nMood: {mood}\nColors: {colors}\nFeedback: {feedback}\n\n")
            f.write(f"VISIONS:\n{ideas}\n\n")
            f.write(f"REFLECTION:\n{critic}\n\n")
            f.write(f"MASTERWORK:\n{final}\n")
        
        print(f"💾 Saved to: {filename}")
        print("\n✨ Masterpiece complete!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()