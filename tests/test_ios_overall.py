from src.ios.overall import process_ios_overall_screenshot

if __name__ == "__main__":
    # test of a dark mode photo 1
    result1 = process_ios_overall_screenshot("data/ios/ios_overall_test.jpg")
    print(result1)

    # test of a dark mode photo 1
    result2 = process_ios_overall_screenshot("data/ios/ios_overall_test2.jpg")
    print(result2)

    # test of a light mode photo
    result3 = process_ios_overall_screenshot("data/ios/ios_overall_test_light.jpg")
    print(result3)