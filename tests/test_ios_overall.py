from src.ios.overall import process_ios_overall_screenshot

if __name__ == "__main__":
    result1 = process_ios_overall_screenshot("data/ios/ios_overall_test.jpg")
    print(result1)

    result2 = process_ios_overall_screenshot("data/ios/ios_overall_test2.jpg")
    print(result2)