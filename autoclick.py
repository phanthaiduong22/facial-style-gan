import pyautogui
import time
 
def click(): 
    print("click")
    time.sleep(300)     
    pyautogui.click()
 
def main():
    while(1):
        click()
 
main()