/*
 * @name	 	scene_services.cpp
 * @brief	 	Provides services to spawn and remove pedestrians dynamically. 
 *          The spawned agents are forwarded to flatland
 * @author 	Ronja Gueldenring
 * @date 		2019/04/05
 **/

#ifndef _scene_service_h_
#define _scene_service_h_

#include <ros/ros.h>
#include <pedsim_simulator/scene.h>
#include <pedsim_srvs/SpawnPeds.h>
#include <pedsim_srvs/SpawnObstacle.h>
#include <pedsim_srvs/MovePeds.h>
#include <flatland_msgs/Model.h>
#include <pedsim_msgs/Ped.h>
#include <pedsim_msgs/LineObstacle.h>
#include <pedsim_msgs/LineObstacles.h>
#include <std_srvs/SetBool.h>

/**
 * This class provides services to spawn and remove pedestrians dynamically.
 */
class SceneServices {
  // Constructor and Destructor
 public:
  SceneServices();
  virtual ~SceneServices() = default;

  ros::ServiceServer respawn_peds_service_;
  ros::ServiceServer remove_all_peds_service_;
  ros::ServiceServer remove_all_peds_behavior_modelling_service_;
  ros::ServiceServer spawn_ped_service_;
  ros::ServiceServer add_obstacle_service_;
  ros::ServiceServer move_peds_service_;
  ros::ServiceServer spawn_peds_service_;
  ros::ServiceServer reset_peds_service_;

  static int agents_index_;

  /**
  * @brief Spawns pedestrian in pedsim and flatland.
  */
  bool spawnPeds(pedsim_srvs::SpawnPeds::Request &request, pedsim_srvs::SpawnPeds::Response &response);

  /**
  * @brief Removes all pedestrians in flatland.
  */
  bool removeAllPeds(std_srvs::SetBool::Request &request, std_srvs::SetBool::Response &response);

  /**
  * @brief Resets all pedestrians to their initial position and state
  */
  bool resetPeds(std_srvs::SetBool::Request &request, std_srvs::SetBool::Response &response);

  /**
  * @brief Spawns shelfes for the forklift.
  */
  bool spawnStaticObstacles(AgentCluster* cluster, std::vector<int> ids);

  /**
  * @brief Respawning means reusing objects from previous tasks.
  * It is a more efficient way to setup a task during learning.
  */
  bool respawnPeds(pedsim_srvs::SpawnPeds::Request &request,
                                pedsim_srvs::SpawnPeds::Response &response);

  bool moveAgentClustersInPedsim(pedsim_srvs::MovePeds::Request &request,
                                pedsim_srvs::MovePeds::Response &response);

  /**
  * @brief Adding static obstacles to pedsim.
  */
  bool addStaticObstacles(pedsim_srvs::SpawnObstacle::Request &request,
                          pedsim_srvs::SpawnObstacle::Response &response);

 protected:
  ros::NodeHandle nh_;

 private:
  /**
  * @brief Removing all pedestrians in pedsim.
  * @return corresponding flatland namespaces of pedestrians
  */
  std::vector<std::string> removePedsInPedsim();

  /**
  * @brief Adding pedestrian to pedsim.
  * @return corresponding AgentCluster
  */
  AgentCluster* addAgentClusterToPedsim(pedsim_msgs::Ped ped, std::vector<int> ids);

  std::vector<flatland_msgs::Model> getFlatlandModelsFromAgentCluster(AgentCluster* agentCluster, std::string yaml_file, std::vector<int> ids);
  std::vector<int> generateAgentIds(int n);

  std::string spawn_models_topic_;
  ros::ServiceClient spawn_models_client_;

  std::string respawn_models_topic_;
  ros::ServiceClient respawn_models_client_;

  std::string delete_models_topic_;
  ros::ServiceClient delete_models_client_;
};

#endif /* _scene_service_h_ */
