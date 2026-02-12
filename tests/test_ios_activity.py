from src.ios.activity import process_ios_category_screenshot

if __name__ == "__main__":
    result1 = process_ios_category_screenshot("data/ios/ios_entertainment_test.jpg")
    print(result1)
    result2 = process_ios_category_screenshot("data/ios/ios_social_test.jpg")
    print(result2)
