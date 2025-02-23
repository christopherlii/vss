import streamlit as st
import cv2
import numpy as np
import os
import torch
import open_clip
from PIL import Image

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
#clip model

@st.cache_resource(show_spinner=False)
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained = 'laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

model, preprocess, tokenizer = load_clip()


#computing cosine similarity with NumPy
def similarity(a, b):
    return np.dot(a,b.T)/(np.linalg.norm(a) * np.linalg.norm(b))


#problem: line 36 is not working, and not saving files to the directories
def get_thumbnail(video_path, thumbnail_path):
    #extracting frist frame from video
    vid = cv2.VideoCapture(video_path)

    success, frame = vid.read()
    if not success:
        raise ValueError("Failed to read vid frame")
    
    #converted frame to rgb
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    Image.fromarray(frameRGB).save(thumbnail_path)
    vid.release()
    return frameRGB



def save_thumbnail(video_path, thumbnail_path, uploaded_file):
    try:
        #st.write("0")
        get_thumbnail(video_path, thumbnail_path)
        #st.write("1")
        
        #process image with CLIP
        image = Image.open(thumbnail_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)

        #st.write("2")

        #generating embedding
        with torch.no_grad():
            embedding = model.encode_image(image_tensor).cpu().numpy()

        #st.write("3")
        st.session_state.thumbnail_db.append({
            'name': uploaded_file.name,
            'video_path': video_path,
            'thumbnail_path': thumbnail_path,
            'embedding': embedding
        })
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}")
        os.remove(video_path)

def main():
    st.write("Clip Image Search")

    if 'thumbnail_db' not in st.session_state:
        st.session_state.thumbnail_db = []

    #make folders for videos and thumbnails
    os.makedirs("videos", exist_ok=True)
    os.makedirs("thumbnails", exist_ok=True)

    curr = []

    with st.expander("Upload videos", expanded=True):
        uploaded_files = st.file_uploader("select video files", type=["mp4", "avi", "mov"], accept_multiple_files=True)

        for uploaded_file in uploaded_files:
            curr.append(uploaded_file.name)
            #check for duplicates
            if any(uploaded_file.name == v['name'] for v in st.session_state.thumbnail_db):
                continue

            #generate path for video
            video_path = os.path.abspath(os.path.join("videos", uploaded_file.name))
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            thumbnail_path = os.path.abspath(os.path.join("thumbnails", f"{uploaded_file.name}.jpg"))
            
            
            save_thumbnail(video_path, thumbnail_path, uploaded_file)
        
        to_remove = []
        for entry in st.session_state.thumbnail_db.copy():
            if entry['name'] not in curr:
                # Remove files from disk
                if os.path.exists(entry['video_path']):
                    os.remove(entry['video_path'])
                if os.path.exists(entry['thumbnail_path']):
                    os.remove(entry['thumbnail_path'])
                # Remove from database
                st.session_state.thumbnail_db.remove(entry)
            

    


    st.divider()
    query_embedding = None
    query = st.text_input("Describe the video you are looking for", placeholder = "e.g. 'a dog playing on the beach'")
    if query:
        text = tokenizer([query])
        with torch.no_grad():
            query_embedding = model.encode_text(text).cpu().numpy()
    
    
    if st.session_state.thumbnail_db:
        if query_embedding is not None:
            # Calculate similarities
            similarities = []
            for video in st.session_state.thumbnail_db:
                sim = similarity(query_embedding, video['embedding'])[0][0]
                similarities.append(sim)
            
            # Sort videos by similarity score
            sorted_videos = sorted(
                zip(st.session_state.thumbnail_db, similarities),
                key=lambda x: x[1], 
                reverse=True
            )

            if len(sorted_videos) == 0:
                st.write("No videos")

        else:
            sorted_videos = [(video, None) for video in st.session_state.thumbnail_db]

        cols = st.columns(3)
        for idx, (video, score) in enumerate(sorted_videos):
            with cols[idx % 3]:
                st.image(
                    video['thumbnail_path'],
                    caption=f"Score: {score:.2f}" if score is not None else video['name'],
                    use_container_width=True
                )
                #st.video(video['video_path'])
                #st.caption(video['name'])





if __name__ == "__main__":
    main()
         