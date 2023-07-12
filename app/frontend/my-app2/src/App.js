import React, { useEffect, useState } from 'react'
import { Stage, Layer, Line, Image, Circle, Group } from 'react-konva'
import axios from 'axios'
import { Menu, MenuContainer } from './components/Menu/Menu'
import { v4 as uuidv4 } from 'uuid'

import './AnnotationArea.css'

axios.defaults.baseURL = 'http://localhost:8000'

const StageContainer = ({ children }) => {
  const style = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flex: 1,
    overflow: 'auto',
  }

  return <div style={style}>{children}</div>
}

async function getImageWithPredictions(imageId, imageType, callback) {
  // If we show the amplitude image, we want to use it for the masks
  const response = await fetch('http://localhost:8000/images', {
    method: 'POST',
    headers: {
      accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image_id: imageId, image_type: imageType }),
  })
  const response_json = await response.json()
  callback(response_json)
  return response_json
}

const AnnotationArea = () => {
  const style = {
    display: 'flex',
    alignItems: 'flex-start', // align items vertically
    justifyContent: 'flex-start', // align items horizontally
    height: '100%', // 100% of the viewport height
    width: '100%', // 100% of the viewport height
  }

  // Image management
  const [amplitudeImage, setAmplitudeImage] = useState(null)
  const [phaseImage, setPhaseImage] = useState(null)
  const [showAmplitudeImage, setShowAmplitudeImage] = useState(true) // 0 for amplitude, 1 for phase
  const [image, setImage] = useState(null)
  const [imageId, setImageId] = useState(0)
  const [currentDataset, setCurrentDataset] = useState(null)
  const img = new window.Image()

  // Polygon management
  const [polygons, setPolygons] = useState([])
  const [polygonCounter, setPolygonCounter] = useState(0)
  const [newPolygonCounter, setnewPolygonCounter] = useState(0)
  const [currentPolygon, setCurrentPolygon] = useState([])
  const currentPolygonRef = React.useRef(currentPolygon)
  const [nextPoint, setNextPoint] = useState(null)

  // Preview line management
  const [previewLine, setPreviewLine] = useState(null)
  const isDrawing = React.useRef(false)

  function getColorByClassId(classId) {
    switch (classId) {
      case 'rbc':
        return '#ff0000'
      case 'wbc':
        return '#ffffff'
      case 'plt':
        return '#0000ff'
      case 'agg':
        return '#00ff00'
      case 'oof':
        return '#ffff00'
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
    const transformedPolygons = []

    polygonsWithPredictions.forEach((polygonWithPrediction, index) => {

      const currentPolygon = []
      for (let i = 0; i < polygonWithPrediction.polygon.points.length; i += 8) {
        currentPolygon.push({
          x: polygonWithPrediction.polygon.points[i] * window.innerWidth,
          y: polygonWithPrediction.polygon.points[i+1] * window.innerHeight,
          color: getColorByClassId(polygonWithPrediction.class_id),
          id: uuidv4()
        })
      }
      transformedPolygons.push(currentPolygon)
      // const polygonPoints = polygonWithPrediction.polygon.points
      // const color = getColorByClassId(polygonWithPrediction.class_id)
      // transformedPolygons.push(polygonPoints)
      // setPolygonCounter(index + 1)
    })
    setPolygons(transformedPolygons)
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
    const image_type = showAmplitudeImage ? 0 : 1
    getImageWithPredictions(imageId, image_type, setImageCallback)
  }, [imageId, showAmplitudeImage, currentDataset])

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
        finishPolygon()
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

  const handleClick = (e) => {
    setCurrentPolygon([
      ...currentPolygon,
      { x: e.evt.x, y: e.evt.y, id: uuidv4() },
    ])
    console.log(currentPolygon)
  }

  const handleMouseMove = (e) => {
    setNextPoint({ x: e.evt.x, y: e.evt.y })
  }

  const finishPolygon = () => {
    if (currentPolygonRef.current.length > 1) {
      console.log(polygons)
      setPolygons((prevPolygons) => [...prevPolygons, currentPolygonRef.current])
    }
    setCurrentPolygon([])
    setNextPoint(null)
  }

  const nextImage = () => {
    setImageId((prevId) => prevId + 1)
  }

  const prevImage = () => {
    setImageId((prevId) => prevId - 1)
  }

  const handleButtonClick = (e) => {
    e.preventDefault()

    const newImageId = e.target.image_id.value
    // Validate the number
    if (newImageId >= 1 && newImageId <= 1000) {
      setImageId(newImageId)
      // Perform your desired action with the valid number
      console.log('Valid number:', newImageId)
    } else {
      console.log('Invalid number:', newImageId)
    }
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
    console.log('Saving masks...')
    const currentPolygonPoints = currentPolygonRef.current[0].points

    if (currentPolygonPoints.length <= 4) {
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
    console.log(polygonCounter)
    console.log(newPolygonCounter)
    const itemstodelete = polygonCounter - newPolygonCounter
    console.log(itemstodelete)
    setPolygons((prev) => {
      const filteredDictionary = Object.entries(prev)
        .filter(([k, v]) => k < itemstodelete) // Change the condition based on your requirement
        .reduce((obj, [k, v]) => {
          obj[k] = v
          return obj
        }, {})

      return filteredDictionary
    })
    setCurrentPolygon([])
    setPolygonCounter(itemstodelete)
    console.log(polygonCounter)
    setnewPolygonCounter(0)
    setPreviewLine(null)
  }

  function oldreset() {
    setPolygons({})
    setCurrentPolygon([])
    setPolygonCounter(0)
    setPreviewLine(null)
  }

  function undo() {
    setPreviewLine(null)
    if (currentPolygonRef.current.points != []) {
      setCurrentPolygon((prev) => {
        const copy = currentPolygonRef.current
        copy[0].points = copy[0].points.slice(0, -4)
        return copy
      }, {})
    }
  }

  function onSegmentationMethodChange(e) {
    const selectedMethod = e.target.value

    axios
      .post('/select_segmentator', {
        method: selectedMethod,
      })
      .then((response) => {
        console.log(response.data) // Output the server's response to the console.
      })
      .catch((error) => {
        console.error(`Error selecting segmentation method: ${error}`)
      })
  }

  function onDatasetChange(e) {
    const selectedDataset = e.target.value

    console.log(`Selected dataset: ${selectedDataset}`)

    axios
      .post('/select_dataset', {
        filename: selectedDataset,
      })
      .then((response) => {
        console.log(response.data) // Output the server's response to the console.
      })
      .catch((error) => {
        console.error(`Error selecting dataset: ${error}`)
      })

    setCurrentDataset(selectedDataset)
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
          onImageId={handleButtonClick}
          onToggleImage={toggleImage}
          onSegmentationMethodChange={onSegmentationMethodChange}
          onDatasetChange={onDatasetChange}
        />
      </MenuContainer>
      <StageContainer>
        <Stage key='main-stage' width={window.innerWidth} height={window.innerHeight} onClick={handleClick} onMouseMove={handleMouseMove}>
          <Layer key='0'>{image && <Image width={window.innerWidth} height={window.innerHeight} image={image} x={0} y={0} />}</Layer>
          <Layer key='1'>
            {polygons.map((polygon, i) => (
              <Group
                key={`group-${i}-${polygon[0].id}`}
                draggable
                onClick={(e) => {
                  e.cancelBubble = true
                  console.log('Clicked on polygon', i)
                }}
                onDragEnd={(e) => {
                  const newPolygon = polygon.map((p) => ({
                    x: p.x + e.target.x(),
                    y: p.y + e.target.y(),
                    color: p.color,
                    id: p.id,
                  }))
                  const newPolygons = [...polygons]
                  newPolygons[i] = newPolygon
                  setPolygons(newPolygons)
                  e.target.position({ x: 0, y: 0 }) // Reset group's position
                }}
              >
                <Line
                  points={polygon.flatMap((p) => [p.x, p.y])}
                  fill={polygon[0].color}
                  opacity={0.5}
                  stroke={polygon[0].color}
                  strokeWidth={4}
                  closed
                />
                {polygon.map((point, j) => (
                  <Circle
                    key={`circle-${j}-${point.id}`}
                    x={point.x}
                    y={point.y}
                    radius={3}
                    fill='#ffff00'
                    draggable
                    onDragEnd={(e) => {
                      e.cancelBubble = true // stop event propagation
                      const newPoint = {
                        x: e.target.x() + e.target.getParent().x(),
                        y: e.target.y() + e.target.getParent().y(),
                        id: point.id,
                      }
                      const newPolygon = [...polygon]
                      newPolygon[j] = newPoint
                      const newPolygons = [...polygons]
                      newPolygons[i] = newPolygon
                      setPolygons(newPolygons)
                    }}
                  />
                ))}
              </Group>
            ))}
            {currentPolygon.length > 0 && (
              <Line
                key='preview'
                points={[
                  ...currentPolygon.flatMap((p) => [p.x, p.y]),
                  nextPoint
                    ? nextPoint.x
                    : currentPolygon[currentPolygon.length - 1].x,
                  nextPoint
                    ? nextPoint.y
                    : currentPolygon[currentPolygon.length - 1].y,
                ]}
                stroke='#000'
                strokeWidth={4}
              />
            )}
          </Layer>
        </Stage>
      </StageContainer>
    </div>
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
