from src.android.activity_history import process_android_activity_history

if __name__ == "__main__":
    # List of images for this person
    images = [
        "data/android/android_activity_test1.jpg",
        "data/android/android_activity_test2.jpg"
    ]

    result = process_android_activity_history(images)
    print(result)
