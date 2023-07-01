import React, { useEffect, useState } from 'react'
import { Stage, Layer, Group, Line, Circle, Image } from 'react-konva'

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
  // callback(`data:image/jpeg;base64,${response_json.amplitude_img_data}`)
  return response_json
}

const AnnotationArea = () => {
  const [amplitudeImage, setAmplitudeImage] = useState(null)
  const [phaseImage, setPhaseImage] = useState(null)
  const [showAmplitudeImage, setShowAmplitudeImage] = useState(true) // 0 for amplitude, 1 for phase
  const [image, setImage] = useState(null)
  const [imageId, setImageId] = useState(0)

  // Polygon management
  const [polygons, setPolygons] = useState({})
  const [polygonCounter, setPolygonCounter] = useState(0)
  const [currentPolygon, setCurrentPolygon] = useState([])
  const currentPolygonRef = React.useRef(currentPolygon)

  const [previewLine, setPreviewLine] = useState(null)
  const isDrawing = React.useRef(false)

  const img = new window.Image()

  function setImageCallback(response_json) {
    // This is a callback function that is called when the image is fetched
    // Its only purpose is to set the image state variables
    
    setAmplitudeImage(`data:image/jpeg;base64,${response_json.amplitude_img_data}`)
    setPhaseImage(`data:image/jpeg;base64,${response_json.phase_img_data}`)

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
        setImageId((prevId) => prevId + 1)
      } else if (event.key === 'ArrowLeft') {
        setImageId((prevId) => prevId - 1)
      } else if (event.key === 't') {
        setShowAmplitudeImage((prev) => !prev)
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
        { points: [pos.x, pos.y, pos.x, pos.y] },
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

  const handleMouseUp = () => {
    // isDrawing.current = false;
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
    <Stage
      width={window.innerWidth}
      height={window.innerHeight}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Layer>
        {image && (
          <Image
            image={image}
            x={0}
            y={0}
            width={window.innerWidth}
            height={window.innerHeight}
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

function Polygon({ id, lines }) {
  return (
    <Group>
      {lines.map((line, i) => (
        <React.Fragment key={'polygon-' + id + '-line-' + i}>
          <Line
            points={line.points}
            stroke='#df4b26'
            strokeWidth={2}
            tension={0.2}
          />
          {
            <Circle
              x={line.points[2]}
              y={line.points[3]}
              radius={5}
              fill='black'
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
