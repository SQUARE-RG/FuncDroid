from hmbot.explorer.llm import ask_uitars
from langchain_core.messages import HumanMessage, AIMessage
from hmbot.explorer.action_parser import action_parser
from hmbot.explorer.prompt import event_llm_prompt
from loguru import logger
import time



def excute_action(action: str, device, current_page, actions_history=None) -> str:
    content=[
        {"type": "text", "text": event_llm_prompt.format(instruction=action)},
        # {"type": "text", "text": "Recent Action History:" + ("\n".join(actions_history) if actions_history else " None")},
        {"type": "text", "text": "Current Page Screenshot:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{current_page.encoded_img}"},
        }
    ]
    response = ask_uitars(content)
    print("=========================excute_action==============================")
    print(response)
    print("====================================================================")
    parsed_output = action_parser.parse_action_output(response, current_page.img.shape[1], current_page.img.shape[0])    
    action_type = parsed_output.get("action")   
    if action_type == "click" and parsed_output.get("point"):
        center_pos = parsed_output["point"]
        logger.debug(f"Executing click at coordinates: {center_pos}")
        device.click(center_pos[0], center_pos[1])
        
    elif action_type == "long_click" and parsed_output.get("point"):
        center_pos = parsed_output["point"]
        logger.debug(f"Executing long_click at coordinates: {center_pos}")
        device.long_click(center_pos[0], center_pos[1])

    elif action_type == "input" and parsed_output.get("content") and parsed_output.get("point"):
        logger.debug(f"Executing input with content: {parsed_output['content']}")
        try:
            center_pos = parsed_output["point"]
            device.click(center_pos[0], center_pos[1])
            time.sleep(1) 
            device.input(parsed_output["content"])
        except Exception as e:
            logger.error(f"Error executing input action: {e}")

    elif action_type == "press_back":
        logger.debug("Executing press_back action")
        device.back()
        
    return parsed_output["point"]