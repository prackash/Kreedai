from flask import Flask, request, jsonify, render_template
import cv2
from ultralytics import YOLO
import numpy as np
import os
import pyodbc
import logging
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# import SQLAlchemy

if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('uploads'):
    os.makedirs('uploads')
app = Flask(__name__)

yhit = []
xhit = []
tframe = []


server= 'localhost,1401'
database = 'Kreedai'
username='SA'
password='RKID#1567'
driver='ODBC Driver 17 for SQL Server'

connection_string = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# connection_string= f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'

# db=SQLAlchemy(app)

# engine=create_engine(connection_string)

# Session=sessionmaker(bind=engine)
# session=Session()

MODEL_PATH = 'model/best.pt'
model = YOLO(MODEL_PATH)

# Function to process frame and draw detections
def process_frame(frame, model):
    results = model(frame)
    for result in results:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            if score > 0.05:  # Confidence threshold
                label = f"{model.names[int(class_id)]} {score:.2f}"
                color = (0, 255, 0)  # Green color for the bounding box
                xhit.append((x1 + x2) / 2)
                yhit.append((y1 + y2) / 2)
                tframe.append(frame.copy())
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Function to process video
def process_video(input_video_path, output_video_path):

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model)
        out.write(frame)

    i = yhit.index(max(yhit))
    cap.release()
    out.release()
    return tframe[i], i

def transform_point(perspective_matrix, x, y):
    point_homogeneous = np.array([x, y, 1])
    transformed_point_homogeneous = np.dot(perspective_matrix, point_homogeneous)
    transformed_point = transformed_point_homogeneous[:2] / transformed_point_homogeneous[2]
    return transformed_point

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files:
        return jsonify({'error':'No video file uploaded'}) , 400
    video = request.files['video'] 
    if video.filename=='':
        return jsonify({'error':'No selected video file'}) , 400
    
    video_path = os.path.join('uploads',video.filename)
    video.save(video_path)
    output_video_path = os.path.join('static','output_video.mp4')

    image, idx = process_video(video_path, output_video_path)
    output_frame_path = 'static/out.jpg'
    cv2.imwrite(output_frame_path, image)
    return jsonify({'message': 'Processing complete', 'frame_index': idx, 'frame_path': output_frame_path})

# engine = create_engine('your_database_url')
@app.route('/warp', methods=['POST'])
def warp():
    frame_index = int(request.form['frame_index'])
    frame_path = request.form['frame_path']
    batter = request.form["batter"]
    bowler_type = request.form["bowler_type"]
    batting_position = request.form["batting_position"]
    innings = request.form["innings"]
    around_the_wicket = request.form["around_the_wicket"]
    match = request.form["match"]
    powerplay = request.form["powerplay"]
    
    try:
        point1 = list(map(int, request.form['point1'].split(',')))
        point2 = list(map(int, request.form['point2'].split(',')))
        point3 = list(map(int, request.form['point3'].split(',')))
        point4 = list(map(int, request.form['point4'].split(',')))
    except ValueError:
        return jsonify({'error': 'Invalid point format'}), 400

    pts = [point1, point2, point3, point4]

    if len(pts) != 4:
        return jsonify({'error': 'Four points are required for warping'}), 400

    logging.info(f"Received points: {pts}")

    img = cv2.imread(frame_path)
    if img is None:
        return jsonify({'error': 'Failed to read the frame image'}), 400

    h, w = img.shape[:2]
    input_pts = np.float32(pts)
    output_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    warp_img = cv2.warpPerspective(img, matrix, (w, h))

    warp_img_path = 'static/warp_img.jpg'
    cv2.imwrite(warp_img_path, warp_img)


    # Draw circle on the original image before transformation
    a = xhit[frame_index]
    b = yhit[frame_index]
    pred_original = cv2.circle(img.copy(), (int(a), int(b)), 3, (255, 0, 0), 4)

    # Transform the circle point to the warped image coordinates
    transformed_point = transform_point(matrix, a, b)
    c, d = int(transformed_point[0]), int(transformed_point[1])
    pred_warped = cv2.circle(warp_img.copy(), (c, d), 3, (255, 0, 0), 4)

    # Save prediction images
    prediction_original_path = "static/prediction_original.jpg"

    cv2.imwrite(prediction_original_path, pred_original)
    pitch_width = 305 # in centimeters
    height,width,_=pred_warped.shape 
    cv2.rectangle(pred_warped, (int((width/2)-((width/pitch_width)*22.86)),0),(int((width/2)+((width/pitch_width)*22.86)),height),(255,0,0),1)
    cv2.rectangle(pred_warped, (int((width/2)-((width/pitch_width)*30.48)),0),(int((width/2)+((width/pitch_width)*30.48)),height),(0,255,0),1)
    cv2.rectangle(pred_warped, (int((width/2)-((width/pitch_width)*61)),0),(int((width/2)+((width/pitch_width)*61)),height),(0,0,255),1)
    # line="onLine"
    # t=0
    # f=int((height/20.2)*2)
    # b=f
    # length=""
    # if(d<t):
    #     length="Full Toss"

    # if (d<b and d>t):
    #     length="Yorker"
        
    # ft=cv2.rectangle(pred_warped,(0,t),(width,b),(100,100,100),1) #full toss
    # t=b
    # b+=f+f

    # if (d<b and d>t):
    #     length="Full"
    # y=cv2.rectangle(ft,(0,t),(width,b),(100,200,100),1)# yorker
    # t=b
    # b+=f

    # if (d<b and d>t):
    #     length="Length"
    # ts=cv2.rectangle(y,(0,t),(width,b),(100,100,200),1)# the slot
    # t=b
    # b+=f

    # if (d<b and d>t):
    #     length="Back of Length"
    # lo=cv2.rectangle(ts,(0,t),(width,b),(200,200,100),1)# length
    # t=b
    # b+=(f/2)

    # if (d<b and d>t):
    #     length="Short"
    # l=cv2.rectangle(lo,(0,t),(width,b),(200,200,200),1) # short
    # t=b
    # b=height

    
    # if (d<b and d>t):
    #     length="Bouncer"
    

    # s=cv2.rectangle(l,(0,t),(width,b),(200,200,200),1) # short
    
    line="onLine"
    t=0
    f=int((height/20.2)*2)
    length=""
    if(d<t):
        length="Full Toss"

    print(f,t)
    b=f
    print(b,t)
    if (d<b and d>t):
        length="Yorker"
        
    y=cv2.rectangle(pred_warped,(0,t),(width,b),(100,100,100),1) #Yorker
    t=b
    b=b+(2*f)
    print(b,t)
    if (d<b and d>t):
        length="Full"

    fu=cv2.rectangle(y,(0,t),(width,b),(100,200,100),1)# Full
    t=b
    b+=f
    print(b,t)
    if (d<b and d>t):
        length="Length"

    le=cv2.rectangle(fu,(0,t),(width,b),(100,100,200),1)# Length
    t=b
    b+=f
    print(b,t)
    if (d<b and d>t):
        length="Back of Length"

    bol=cv2.rectangle(le,(0,t),(width,b),(100,100,200),1)# Length
    t=b
    b+=f
    print(b,t)
    if (d<b and d>t):
        length="Short"


    sh=cv2.rectangle(bol,(0,t),(width,b),(200,200,100),1)# short
    t=b
    b=height
    print(b,t)
    if (d<b and d>t):
        length="Bouncer"

    bo=cv2.rectangle(sh,(0,t),(width,b),(200,200,200),1) # bouncer


    if(batting_position):
        if (c<int((width/2)-((width/pitch_width)*22.86))):
            if(c<int((width/2)-((width/pitch_width)*61))):
                line="Off Wide"
            else:
                line="Off"
        elif (c>int((width/2)+((width/pitch_width)*22.86))):
            if(c>int((width/2)+((width/pitch_width)*30.48))):
                line="Leg Wide"
            else:
                line="Leg"
    else:
        if (c>int((width/2)+((width/pitch_width)*22.86))):
            if(c>int((width/2)+((width/pitch_width)*61))):
                line="Off Wide"
            else:
                line="Off"
        elif (c<int((width/2)-((width/pitch_width)*22.86))):
            if(c<int((width/2)-((width/pitch_width)*30.48))):
                line="Leg Wide"
            else:
                line="Leg"
    


    

    # sql_command = f"""
    # SELECT *
    # FROM [Kreedai].[dbo].[data]
    # where Batter like '%Marcus Campopiano%'
    # and [Bowler_Type] ='RFM' --choose from LOB,RFM,LM,RM,LC,RLB,SLA,RF,LF,ROB,LFM
    # and Line ='Off' --choose from Leg Wide,Off Wide,Line,Leg,Off
    # and Length ='Full' --Choose from Yorker,Back Of Length,Length,Full,Short,Full Toss,Bouncer
    
    # --OPTIONAL SETTINGS
    # --and Innings = --Choose from 1 or 2
    # --and [Power Play] = --Choose from 1,2,3 or Null
    # --and venue = --Choose from list of all possible venues for the batter
    # order by [Date] desc, innings desc, [Over] desc, ball desc
    # """

    sql_command = f"""
    with cte as 
    (
    SELECT Date,Batter,Shot, FieldX,FieldY,[Events],Runs,Wicket,ROW_NUMBER() over (order by [Date] desc, innings desc, [Over] desc, ball desc) as [Row]
    FROM [Kreedai].[dbo].[data]
    where Batter like '%Marcus Campopiano%'
    and [Bowler_Type] = '{bowler_type}' --choose from LOB,RFM,LM,RM,LC,RLB,SLA,RF,LF,ROB,LFM
    and Line = '{line}' --choose from Leg Wide,Off Wide,Line,Leg,Off
    and Length = '{length}' --Choose from Yorker,Back Of Length,Length,Full,Short,Full Toss,Bouncer
    
    --OPTIONAL SETTINGS
    --and Innings = --Choose from 1 or 2
    --and [Power Play] = --Choose from 1,2,3 or Null
    --and venue = --Choose from list of all possible venues for the batter
    --and Match = --Choose from list of all possible matches for the batter

    --FILTERING
    and Shot <> ''

    )

    select * from cte
    where Row = 1
    """


    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()
    cursor.execute(sql_command)
    data=cursor.fetchall()
    row=[]
    for r in data:
        print(r)
        row.append(r)
        print('\n')
    data=str(data)
    print(row)
    column_names=['Runs','Shot','Wicket','FieldX','FieldY']
    rows_dict = [dict(zip(column_names,ro)) for ro in row]
    print(rows_dict)

    prediction_original_path = "static/prediction_original.jpg"
    prediction_warped_path = "static/prediction_warped.jpg"
    cv2.imwrite(prediction_original_path, pred_original)
    cv2.imwrite(prediction_warped_path, bo)


    return jsonify({
        'warp_img_path': warp_img_path,
        'prediction_original_path': prediction_original_path,
        'prediction_warped_path': prediction_warped_path,
        "length":length,
        "line":line,
        'data':data,
        'rows_dict':rows_dict
    })
    # result = db.engine.execute(sql_command)
    # rows = result.fetchall()
    
    # data = []
    # for row in rows:
    #     data.append(dict(row.items()))    

    # result = session.execute(sql_command)
    # data=result.fetchall()
    # data=str(data)




    # conn = pyodbc.connect(connection_string)
    # cursor = conn.cursor()
    # cursor.execute("SELECT * FROM table")
    # rows = cursor.fetchall()

    # result = []
    # for row in rows:
    #     result.append(dict(zip([column[0] for column in cursor.description],row)))
    # conn.close()
    

    # sql_query = request.json.get('sql_command')
    # try:
    #     # Execute the SQL query using SQLAlchemy
    #     result = engine.execute(sql_query)
    #     # Fetch all rows from the result
    #     rows = result.fetchall()
    #     # Convert rows to a list of dictionaries
    #     data = [dict(row) for row in rows]
    #     # Close the result proxy
    #     result.close()

    #     # Return the query result as JSON response
    #     return jsonify({'success': True, 'data': data})
    # except Exception as e:
    #     return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
