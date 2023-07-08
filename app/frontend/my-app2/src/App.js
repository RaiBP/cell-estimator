import React, { useEffect, useState } from 'react'
import { Stage, Layer, Group, Line, Circle, Image } from 'react-konva'

const MenuContainer = ({ children }) => {
  const style = {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'flex-start', // align items to start
    alignItems: 'center',
    background: '#3C3E45',
    width: '10%',
    padding: '20px',
    boxSizing: 'border-box',
    borderRight: '1px solid #eee',
  }

  return <div style={style}>{children}</div>
}

const Button = ({ children, onClick }) => {
  const style = {
    display: 'block',
    padding: '20px 20px',
    marginBottom: '10px',
    backgroundColor: '#5A5F6C',
    color: '#FFF',
    borderRadius: '4px',
    textAlign: 'center',
    cursor: 'pointer',
    width: '80%',
    height: '7%',
    fontSize: '150%'
  }

  return (
    <button onClick={onClick} style={style}>
      {children}
    </button>
  )
}

const StageContainer = ({ children }) => {
  const style = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flex: 1,
    overflow: 'auto',
    background: '#f0f0f0',
  }

  return <div style={style}>{children}</div>
}

function Menu({ onReset, onUndo, onSave, onNext, onPrev, onToggleImage }) {
  return (
    <div
      style={{
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
        width: '10%',
        background: '#f0f0f0',
        padding: '0px',
        
      }}
    >
      <Button onClick={onReset}>Reset</Button>
      <Button onClick={onUndo}>Undo</Button>
      <Button onClick={onSave}>Save</Button>
      <Button onClick={onNext}>Next Image</Button>
      <Button onClick={onPrev}>Previous Image</Button>
      <Button onClick={onToggleImage}>Toggle Image</Button>
    </div>
  )
}

function ImageAnnotation({
  image,
  polygons,
  currentPolygon,
  previewLine,
  onMouseDown,
  onMouseMove,
}) {
  return (
    <Stage
      width={window.innerWidth*0.9}
      height={window.innerHeight*0.9}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
    >
      <Layer>
        {image && (
          <Image
            image={image}
            x={0}
            y={0}
            width={window.innerWidth*0.9}
            height={window.innerHeight*0.9}
          />
        )}
        {Object.entries(polygons).map(([polygonId, polygon]) => {
          return <Polygon key={polygonId} id={polygonId} lines={polygon} />
        })}
        <Polygon key='current' id='current' lines={currentPolygon} />
        {previewLine && (
          <Line
            points={previewLine.points}
            stroke='#df4b26'
            strokeWidth={5}
            tension={0.5}
            lineCap='round'
            lineJoin='round'
          />
        )}
      </Layer>
    </Stage>
  )
}

async function getImageWithPredictions(imageId, callback) {
  const response = await fetch('http://localhost:8000/images', {
    method: 'POST',
    headers: {
      accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image_id: imageId }),
  })
  const response_json = await response.json()
  console.log(response_json)
  callback(response_json)
  return response_json
}

const AnnotationArea = () => {
  const style = {
    display: 'flex',
    alignItems: 'flex-start', // align items vertically
    justifyContent: 'flex-start', // align items horizontally
    height: '100vh', // 100% of the viewport height
  }

  // Image management
  const [amplitudeImage, setAmplitudeImage] = useState(null)
  const [phaseImage, setPhaseImage] = useState(null)
  const [showAmplitudeImage, setShowAmplitudeImage] = useState(true) // 0 for amplitude, 1 for phase
  const [image, setImage] = useState(null)
  const [imageId, setImageId] = useState(0)
  const img = new window.Image()

  // Polygon management
  const [polygons, setPolygons] = useState({})
  const [polygonCounter, setPolygonCounter] = useState(0)
  const [currentPolygon, setCurrentPolygon] = useState([])
  const currentPolygonRef = React.useRef(currentPolygon)

  // Preview line management
  const [previewLine, setPreviewLine] = useState(null)
  const isDrawing = React.useRef(false)

  function getColorByClassId(classId) {
    switch(classId) {
      case 'rbc':
        return '#ff0000';
      case 'wbc':
        return '#ffffff';
      case 'plt':
        return '#0000ff';
      case 'agg':
        return '#00ff00';
      case 'oof':
        return '#ffff00';
    }
  }

  function setImageCallback(response_json) {
    // This is a callback function that is called when the image is fetched
    // Its only purpose is to set the image state variables

    setAmplitudeImage(
      `data:image/jpeg;base64,${response_json.amplitude_img_data}`
    )
    setPhaseImage(`data:image/jpeg;base64,${response_json.phase_img_data}`)

    const polygonsWithPredictions = response_json.predictions
    const transformedPolygons = {}
    polygonsWithPredictions.forEach((polygonWithPrediction, index) => {
      const polygon = polygonWithPrediction.polygon.points
      const color = getColorByClassId(polygonWithPrediction.class_id)
      const transformedPolygon = []
      for (let i = 0; i < polygon.length; i += 2) {
        transformedPolygon.push({
          points: [
            polygon[i],
            polygon[i + 1],
            polygon[i + 2],
            polygon[i + 3],
          ],
          color
        })
      }
      transformedPolygons[index] = transformedPolygon
      setPolygonCounter(index+1)
    })
    setPolygons(transformedPolygons)

    console.log(response_json.predictions)
  }

  // Hook for showing amplitude or phase image
  useEffect(() => {
    if (showAmplitudeImage) {
      img.src = amplitudeImage
    } else {
      img.src = phaseImage
    }
    img.onload = () => {
      setImage(img)
    }
  }, [showAmplitudeImage, amplitudeImage, phaseImage])

  useEffect(() => {
    getImageWithPredictions(imageId, setImageCallback)
  }, [imageId])

  // Hook for keeping track of lines
  useEffect(() => {
    currentPolygonRef.current = currentPolygon
  }, [currentPolygon])

  // Hook for registering keydown events -- happens only when component is mounted
  useEffect(() => {
    // Handling keydown events -- registering callback
    const handleKeyDown = (event) => {
      if (event.key === 'r' && event.ctrlKey) {
        reset()
      }
      if (event.key === 'r') {
        resetCurrentPolygon()
      } else if (event.key === 'z' && event.ctrlKey) {
        undo()
      } else if (event.key === 'Escape') {
        stopDrawing()
      } else if (event.key === 's') {
        saveMask()
      } else if (event.key === 'ArrowRight') {
        nextImage()
      } else if (event.key === 'ArrowLeft') {
        prevImage()
      } else if (event.key === 't') {
        toggleImage()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  const handleMouseDown = (e) => {
    isDrawing.current = true
    if (previewLine) {
      setCurrentPolygon([...currentPolygon, previewLine])
      setPreviewLine(null)
    } else {
      const pos = e.target.getStage().getPointerPosition()
      setCurrentPolygon([
        ...currentPolygon,
        { points: [pos.x, pos.y, pos.x, pos.y], color:'#bfff00' },
      ])
    }
  }

  const handleMouseMove = (e) => {
    const stage = e.target.getStage()
    const point = stage.getPointerPosition()

    if (!isDrawing.current || currentPolygon.length === 0) {
      return
    }

    const lastLineEnd =
      currentPolygon[currentPolygon.length - 1].points.slice(2)

    // Computing current distance to first point
    const firstPoint = currentPolygon[0].points
    const dx = firstPoint[0] - point.x
    const dy = firstPoint[1] - point.y
    const distance = Math.sqrt(dx * dx + dy * dy)
    if (distance < 15) {
      const firstPoint = currentPolygon[0].points
      setPreviewLine({ points: [...lastLineEnd, ...firstPoint] })
    } else {
      setPreviewLine({ points: [...lastLineEnd, point.x, point.y] })
    }
  }

  const nextImage = () => {
    setImageId((prevId) => prevId + 1)
  }

  const prevImage = () => {
    setImageId((prevId) => prevId - 1)
  }

  const toggleImage = () => {
    setShowAmplitudeImage((prev) => !prev)
  }

  function stopDrawing() {
    saveMask()
    isDrawing.current = false
    setPreviewLine(null)
  }

  function saveMask() {
    if (currentPolygonRef.current.length <= 2) {
      resetCurrentPolygon()
      return
    }

    setPolygonCounter((prevCount) => {
      setPolygons((prevPolygons) => {
        return {
          ...prevPolygons,
          [prevCount]: currentPolygonRef.current.slice(0),
        }
      })
      return prevCount + 1
    })
    setCurrentPolygon([])
  }

  function resetCurrentPolygon() {
    setCurrentPolygon([])
    setPreviewLine(null)
  }

  function reset() {
    setPolygons({})
    setCurrentPolygon([])
    setPolygonCounter(0)
    setPreviewLine(null)
  }

  function undo() {
    setPreviewLine(null)
    setCurrentPolygon((lines) => lines.slice(0, -1))
  }

  return (
    <div style={style}>
      <MenuContainer>
        <Menu
          onReset={reset}
          onUndo={undo}
          onSave={saveMask}
          onNext={nextImage}
          onPrev={prevImage}
          onToggleImage={toggleImage}
        />
      </MenuContainer>
      <StageContainer>
        <ImageAnnotation
          image={image}
          polygons={polygons}
          currentPolygon={currentPolygon}
          previewLine={previewLine}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
        />
      </StageContainer>
    </div>
  )
}

const Polygon = (props) => {
  return (
    <Group>
      {props.lines.map((line, i) => (
        <React.Fragment key={'polygon-' + props.id + '-line-' + i}>
          <Line
            points={line.points.map((p, index) => {
              // Rescale the points to the current canvas size
              return index % 2 === 0
                ? p * window.innerWidth*0.9
                : p * window.innerHeight*0.9
            })}
            stroke={line.color}
            strokeWidth={3}
            tension={0.2}
          />
          {
            <Circle
              x={line.points[2]}
              y={line.points[3]}
              radius={5}
              fill='green'
            />
          }
        </React.Fragment>
      ))}
    </Group>
  )
}

class App extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      counter: 0,
      image: null,
    }
  }

  render() {
    return (
      <div className='App'>
        <div className='App-header'>
          <AnnotationArea />
        </div>
      </div>
    )
  }
}

export default App
