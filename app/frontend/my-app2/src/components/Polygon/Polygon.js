import React from 'react'
import { Group, Line, Circle } from 'react-konva'

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

export default Polygon
