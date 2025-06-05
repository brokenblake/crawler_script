from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os

# 设置 ChromeDriver 的路径

driver = webdriver.Edge()
def download_audio(link, keywords):
    # 处理关键词，创建一个安全的文件夹名
    safe_keywords = ''.join(c for c in keywords if c.isalnum() or c in (' ', '_')).rstrip()
    folder_name = safe_keywords
    base_directory="Late Romantic and Impressionist"
    target_directory=os.path.join(base_directory,folder_name)
    os.makedirs(target_directory, exist_ok=True)  # 确保文件夹存在
    
    # 下载音频文件
    response = requests.get(link,proxies=proxies)
    file_name = os.path.join(target_directory, link.split('/')[-1])  # 使用变量值作为文件夹名称

    with open(file_name, 'wb') as f:
        f.write(response.content)
    print(f"下载音频文件: {file_name}")
    
    file_path = os.path.join(target_directory, "File")
    if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
        os.remove(file_path)
        print(f"删除空文件: {file_path}")

def write_with_wrap(text, file, max_length=50):
    # 先按换行符分割文本
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        words = paragraph.split()  # 按空格分割成单词
        current_line = ""

        for word in words:
            # 如果添加新单词后超过最大长度，则换行
            if len(current_line) + len(word) + 1 > max_length:
                file.write(current_line + "\n")  # 写入当前行并换行
                current_line = word  # 重置当前行，开始新的一行
            else:
                current_line += (" " + word) if current_line else word  # 添加单词到当前行

        if current_line:  # 写入最后一行
            file.write(current_line)
        
        file.write("\n")  # 在段落结束后添加一个空行

try:
    # 打开网站
    driver.get("https://imslp.org/")

    # 等待页面加载
    time.sleep(3)

    # 找到搜索框
    search_box = driver.find_element(By.ID, 'searchInput')  # 根据实际情况调整

    # 输入搜索关键词
    search_query = "Danse Macabre, Op. 40"
    safe_keywords = ''.join(c for c in search_query if c.isalnum() or c in (' ', '_')).rstrip()

    query_words = search_query.split()
    
    print(query_words)
    search_box.send_keys(search_query)

    # 提交搜索
    search_box.send_keys(Keys.RETURN)

    # 等待搜索结果加载
    WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'h3')))

    # 获取并打印搜索结果
    results = driver.find_elements(By.CSS_SELECTOR, 'h3 ')  # 根据实际情况调整
    for result in results:
        if any(word in result.text for word in query_words): 
            print(f"找到符合条件的结果: {result.text}")
            result.click()
            time.sleep(3)
            driver.switch_to.window(driver.window_handles[-1])
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'wiki-body')))
            
            print(f"当前页面 URL: {driver.current_url}")
            text_elements = driver.find_elements(By.CSS_SELECTOR, '.wi_body')
            filtered_text = []
            text_head = driver.find_element(By.ID, 'firstHeading')
            filtered_text=[text_head.text]
            try:
                text_head_sub = driver.find_element(By.ID, 'contentSub')
                filtered_text.append(text_head_sub.text)
            except :
                print("没有找到副标题")
                
            filtered_text +=  [element.text for element in text_elements]
     
            base_directory = "Late Romantic and Impressionist"
            target_directory = os.path.join(base_directory, safe_keywords)
            os.makedirs(target_directory, exist_ok=True)  # 确保文件夹存在
            text_file_name = os.path.join(target_directory, f"{safe_keywords}.txt")
          
            with open(text_file_name, 'w', encoding='utf-8') as f:
                for text in filtered_text:
                   write_with_wrap(text, f, max_length=100)

            print(f"保存文本文件: {text_file_name}")
            try:
                we_element = driver.find_element(By.CLASS_NAME, "we")
                audio_elements = we_element.find_elements(By.CSS_SELECTOR,'a[href$=".mp3"], a[href$=".ogg"], a[href$=".wav"], a[href$=".FLAC"],a[href$=".flac"]')  # 选择 href 属性以 .mp3 结尾的 <a> 标签
                mp3_links=[element.get_attribute("href") for element in audio_elements]
                for index, link in enumerate(mp3_links, start=1):
                    print(f"第 {index} 个音乐文件链接: {link}")
                relevant_links = [link for link in mp3_links ]
                for link in relevant_links:
                    print(f'下载链接: {link}')
                    download_audio(link,search_query)
                break
            finally:
                driver.close()
finally:
    driver.quit()