
import Button from '../Button/Button'
import './ExplainMenu.css'


function ExplainMenu({handleOptionClick}){
    
    return(
        <div className="explain-menu" style={{ left: '${x}px', top: '${y}px'}}>
            <Button className="explain-menu-button" onClick={() => handleOptionClick('close')}>Close</Button>
        </div>) 
}

export{ ExplainMenu }