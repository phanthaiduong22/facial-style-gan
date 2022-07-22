import pyautogui
import time
 
def click(): 
    print("click after 10min")
    time.sleep(10*60)     
    pyautogui.click()
 
def main():
    while(1):
        click()
 
main()