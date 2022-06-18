import pyautogui
import time
 
def click(): 
    time.sleep(5)     
    pyautogui.click()
 
def main():
    for i in range(20): 
        click()
 
main()