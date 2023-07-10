import { wait } from '@testing-library/user-event/dist/utils'
import React, { useEffect, useState } from 'react'
import { Stage, Layer, Group, Line, Circle, Image } from 'react-konva'
import axios from 'axios'

axios.defaults.baseURL = 'http://localhost:8000'

function DatasetSelector({ onChange }) {
  const [datasets, setDatasets] = useState([])

  useEffect(() => {
    async function fetchData() {
      const response = await axios.get('/datasets')
      setDatasets(response.data.datasets)
    }
    fetchData()
  }, [])

  return (
    <div>
      <label for='dataset-selector'>Choose a dataset: </label>
      <select id='dataset-selector' onChange={onChange}>
        {datasets.map((method, index) => (
          <option key={index} value={method}>
            {method}
          </option>
        ))}
      </select>
    </div>
  )
}

function SegmentationMethodsSelector({ onChange }) {
  const [segmentationMethods, setSegmentationMethods] = useState([])

  useEffect(() => {
    async function fetchData() {
      const response = await axios.get('/get_segmentation_methods')
      setSegmentationMethods(response.data.segmentation_methods)
    }
    fetchData()
  }, [])

  return (
    <div>
      <label for='segmentation'>Choose a segmentation method:</label>
      <select id='segmentation' onChange={onChange}>
        {segmentationMethods.map((method, index) => (
          <option key={index} value={method}>
            {method}
          </option>
        ))}
      </select>
    </div>
  )
}

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
    fontSize: '100%',
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

function Menu({
  onReset,
  onUndo,
  onSave,
  onNext,
  onPrev,
  onImageId,
  onToggleImage,
  onSegmentationMethodChange,
  onDatasetChange,
}) {
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
      <form onSubmit={onImageId}>
        <label>
          Enter a number between 1 and 1000:
          <input name='image_id' type='number' />
        </label>
        <input type='submit' value='Submit' />
      </form>
      <SegmentationMethodsSelector onChange={onSegmentationMethodChange} />
      <DatasetSelector onChange={onDatasetChange} />
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
      width={window.innerWidth}
      height={window.innerHeight}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
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
            closed={true}
          />
        )}
      </Layer>
    </Stage>
  )
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
    height: '100vh', // 100% of the viewport height
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
  const [polygons, setPolygons] = useState({})
  const [polygonCounter, setPolygonCounter] = useState(0)
  const [currentPolygon, setCurrentPolygon] = useState([])
  const currentPolygonRef = React.useRef(currentPolygon)

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
    const transformedPolygons = {}
    polygonsWithPredictions.forEach((polygonWithPrediction, index) => {
      const polygonPoints = polygonWithPrediction.polygon.points
      const color = getColorByClassId(polygonWithPrediction.class_id)
      transformedPolygons[index] = [
        {
          points: polygonPoints,
          color,
        },
      ]
      setPolygonCounter(index + 1)
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
    const pos = e.target.getStage().getPointerPosition()
    if (currentPolygon.length === 0) {
      // If there's no polygon started, start a new one
      setCurrentPolygon((prev) => [
        ...prev,
        {
          points: [
            pos.x / window.innerWidth,
            pos.y / window.innerHeight,
            pos.x / window.innerWidth,
            pos.y / window.innerHeight,
          ],
          color: '#ff0000',
        },
      ])
    } else {
      // If a polygon has been started, add a new point to the last polygon
      setCurrentPolygon((prev) => {
        const newPrev = [...prev]
        const lastPolygon = newPrev[newPrev.length - 1]
        lastPolygon.points.push(
          pos.x / window.innerWidth,
          pos.y / window.innerHeight
        )
        return newPrev
      })
    }
  }

  const handleMouseMove = (e) => {
    if (!isDrawing.current) {
      return
    }

    const pos = e.target.getStage().getPointerPosition()
    console.log(currentPolygonRef.current)
    const lastPolygon =
      currentPolygonRef.current[currentPolygonRef.current.length - 1]
    console.log(lastPolygon)
    lastPolygon.points[lastPolygon.points.length - 2] =
      pos.x / window.innerWidth
    lastPolygon.points[lastPolygon.points.length - 1] =
      pos.y / window.innerHeight
    setCurrentPolygon((prev) => [...prev])
  }

  const nextImage = () => {
    setImageId((prevId) => prevId + 1)
  }

  const prevImage = () => {
    setImageId((prevId) => prevId - 1)
  }

  const skip100Images = () => {
    setImageId((prevId) => prevId + 100)
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
    setPolygons({})
    setCurrentPolygon([])
    setPolygonCounter(0)
    setPreviewLine(null)
  }

  function undo() {
    setPreviewLine(null)
    setCurrentPolygon((lines) => lines.slice(0, -1))
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
                ? p * window.innerWidth
                : p * window.innerHeight
            })}
            stroke={line.color}
            strokeWidth={3}
            tension={0.2}
          />
          {
            <Circle
              x={line.points[2] * window.innerWidth}
              y={line.points[3] * window.innerHeight}
              radius={1.0}
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
