from language_tool_python import LanguageTool
import time
# Create a LanguageTool instance
tool = LanguageTool('en-US')  # Specify the language ('en-US' for US English)

# Example text to check
text ="aha"


# Correct the text and print the corrected version
start_time = time.time()
for i in range(0, 100):
    corrected_text = tool.correct(text)
    print("Corrected text:", corrected_text)
print("--- %s seconds ---" % (time.time() - start_time))